import base64
import io
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from classical_pipeline.fovea_localization import estimate_fov_mask
from classical_pipeline.localization_api import find_fovea, find_optic_disk
from dl_pipeline.model import get_model
from dl_pipeline.postprocess import refine_vessel_mask
from dl_pipeline.predict import predict_vessel_mask
from matching.matcher import extract_identity_template, match_templates_detailed, verify_identity


# --- Core Pipeline Service Manager ---
class PipelineService:
    def __init__(self, model_checkpoint: str | Path = "models/best_unet_model_v3.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        project_root = Path(__file__).resolve().parent.parent
        checkpoint_path = project_root / model_checkpoint if not Path(model_checkpoint).is_absolute() else Path(model_checkpoint)
        if not checkpoint_path.exists():
            for fallback_name in ["best_unet_model_v3.pth", "best_unet_model_v2.pth", "best_unet_model.pth"]:
                fallback = project_root / "models" / fallback_name
                if fallback.exists():
                    checkpoint_path = fallback
                    break

        encoder_weights = None if checkpoint_path.exists() else "imagenet"
        self.model = get_model(encoder_weights=encoder_weights).to(self.device)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
        self.model.eval()

    @staticmethod
    def _safe_fov_mask(bgr_image: np.ndarray) -> np.ndarray:
        try:
            return estimate_fov_mask(bgr_image)
        except Exception:
            h, w = bgr_image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            cy, cx = h // 2, w // 2
            rad = int(round(0.475 * min(h, w)))
            cv2.circle(mask.view(np.uint8), (cx, cy), rad, 1, thickness=-1)
            return mask > 0

    @staticmethod
    def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
        if len(raw_bytes) == 0:
            raise ValueError("Image payload is empty.")
        if len(raw_bytes) > 95_000_000:
            raise ValueError("Image payload exceeds maximum allowed size.")

        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if img is None:
            try:
                Image.MAX_IMAGE_PIXELS = 64_000_000
                pil_img = Image.open(io.BytesIO(raw_bytes))
                if pil_img.mode in ("RGBA", "LA") or ("transparency" in pil_img.info):
                    bg = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
                    bg.alpha_composite(pil_img.convert("RGBA"))
                    rgb_arr = np.array(bg.convert("RGB"))
                else:
                    rgb_arr = np.array(pil_img.convert("RGB"))
                img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Failed to decode image buffer: {e}")

        if img.dtype == np.uint16:
            max_val = float(np.max(img))
            scale_factor = 257.0 if max_val > 4095 else (max(1.0, max_val) / 255.0)
            img = np.clip(img.astype(np.float32) / scale_factor, 0, 255).astype(np.uint8)
        elif np.issubdtype(img.dtype, np.floating):
            max_val = float(np.max(img))
            multiplier = 255.0 if max_val <= 1.0 else 1.0
            img = np.clip(img.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            img = np.clip(img[:, :, :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        if h < 32 or w < 32:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small; minimum required is 32x32.")
        if h > 12000 or w > 12000 or (h * w) > 64_000_000:
            raise ValueError(f"Image dimensions ({w}x{h}) exceed maximum allowed 64MP limit.")

        return img

    @staticmethod
    def decode_base64_image(image_base64: str) -> np.ndarray:
        if len(image_base64) > 130_000_000:
            raise ValueError("Image payload exceeds maximum allowed size.")
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_base64)
        return PipelineService.decode_image_bytes(raw_bytes)

    @staticmethod
    def encode_base64_image(image_np: np.ndarray, format_ext: str = ".png") -> str:
        success, buffer = cv2.imencode(format_ext, image_np)
        if not success:
            raise ValueError("Failed to encode image to buffer.")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/png" if format_ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64_str}"

    def normalize_image_for_studio(
        self, raw_bytes: bytes, max_dim: int = 768, filename: str = ""
    ) -> dict[str, Any]:
        img_bgr = self.decode_image_bytes(raw_bytes)
        orig_h, orig_w = img_bgr.shape[:2]

        if max(orig_h, orig_w) > 1000:
            scale = float(max_dim) / float(max(orig_h, orig_w))
            target_w = max(1, int(round(orig_w * scale)))
            target_h = max(1, int(round(orig_h * scale)))
            img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        h, w = img_bgr.shape[:2]
        return {
            "image_base64": self.encode_base64_image(img_bgr, ".png"),
            "width": int(w),
            "height": int(h),
            "original_width": int(orig_w),
            "original_height": int(orig_h),
            "filename": filename,
        }

    @staticmethod
    def apply_perturbation(
        img_bgr: np.ndarray,
        rotate_deg: float = 0.0,
        scale_factor: float = 1.0,
        brightness: float = 1.0,
        blur_sigma: float = 0.0,
        haze_factor: float = 0.0,
        glare_intensity: float = 0.0,
        noise_sigma: float = 0.0,
        occlusion_frac: float = 0.0,
        gamma: float = 1.0,
        spherical_warp: float = 0.0,
        elastic_alpha: float = 0.0,
    ) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        out = img_bgr.astype(np.float32)
        fov = PipelineService._safe_fov_mask(img_bgr) > 0

        # --- Gamma Tone-Curve Adjustment ---
        if abs(gamma - 1.0) > 1e-3:
            out = np.power(np.clip(out / 255.0, 0.0, 1.0), gamma) * 255.0

        # --- Cataract & Media Opacity Haze ---
        if haze_factor > 1e-3:
            veil = np.zeros_like(out)
            veil[:, :, 0] = 125.0
            veil[:, :, 1] = 165.0
            veil[:, :, 2] = 210.0
            blurred_bg = cv2.GaussianBlur(out, (0, 0), sigmaX=14.0)
            haze_layer = 0.65 * veil + 0.35 * blurred_bg
            out = (1.0 - haze_factor) * out + haze_factor * haze_layer

        # --- Corneal Specular Flash Glare ---
        if glare_intensity > 1e-3:
            cy, cx = h / 2.0, w / 2.0
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            glare_radius = 0.22 * min(h, w)
            glare_profile = np.exp(-0.5 * (dist_from_center / (glare_radius * 0.45)) ** 2)
            glare_profile = (glare_profile * glare_intensity * 255.0).astype(np.float32)
            for c in range(3):
                out[:, :, c] = np.clip(out[:, :, c] + glare_profile, 0.0, 255.0)

        # --- Illumination & Flash Brightness ---
        if abs(brightness - 1.0) > 1e-3:
            out = np.clip(out * brightness, 0.0, 255.0)

        # --- Sensor Thermal / Shot Noise ---
        if noise_sigma > 1e-3:
            rng = np.random.default_rng(42)
            noise = rng.normal(0.0, noise_sigma, size=(h, w, 3)).astype(np.float32)
            out = np.clip(out + noise, 0.0, 255.0)

        # --- Optical Defocus Blur ---
        if blur_sigma > 1e-3:
            ksize = int(math.ceil(blur_sigma * 3.0)) * 2 + 1
            out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=blur_sigma)

        # --- Anatomical Eyelid & Eyelash Occlusion ---
        if occlusion_frac > 1e-3:
            occ_h = int(h * occlusion_frac)
            if occ_h > 0:
                out[:occ_h, :] = 0.0
                feather = min(20, occ_h)
                if feather > 1:
                    ramp = np.linspace(0.0, 1.0, feather).reshape(-1, 1, 1).astype(np.float32)
                    out[occ_h - feather : occ_h, :] *= ramp

        # --- Radial Spherical Distortion ---
        if abs(spherical_warp) > 1e-3:
            cx, cy = w / 2.0, h / 2.0
            r_max = math.hypot(cx, cy)
            y_indices, x_indices = np.indices((h, w), dtype=np.float32)
            dx = x_indices - cx
            dy = y_indices - cy
            r_norm = np.sqrt(dx**2 + dy**2) / r_max
            distortion_factor = 1.0 + spherical_warp * (r_norm**2)
            map_x = (cx + dx * distortion_factor).astype(np.float32)
            map_y = (cy + dy * distortion_factor).astype(np.float32)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # --- Elastic Displacement Field ---
        if elastic_alpha > 1e-3:
            rng = np.random.default_rng(42)
            rand_dx = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
            rand_dy = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
            smooth_dx = cv2.GaussianBlur(rand_dx, (31, 31), 6.0)
            smooth_dy = cv2.GaussianBlur(rand_dy, (31, 31), 6.0)
            norm_x = np.max(np.abs(smooth_dx)) or 1.0
            norm_y = np.max(np.abs(smooth_dy)) or 1.0
            smooth_dx = (smooth_dx / norm_x) * elastic_alpha
            smooth_dy = (smooth_dy / norm_y) * elastic_alpha
            y_grid, x_grid = np.indices((h, w), dtype=np.float32)
            map_x = (x_grid + smooth_dx).astype(np.float32)
            map_y = (y_grid + smooth_dy).astype(np.float32)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # --- Rigid Affine Rotation and Scaling ---
        if abs(rotate_deg) > 1e-3 or abs(scale_factor - 1.0) > 1e-3:
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, rotate_deg, scale_factor)
            out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return out

    def segment_vessels(
        self,
        bgr_image: np.ndarray,
        threshold: float = 0.60,
        apply_postprocess: bool = True,
        min_size: int = 25,
        max_hole_size: int = 15,
        use_tta: bool = True,
        compute_skeleton: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, float]], float]:
        t0 = time.perf_counter()
        orig_h, orig_w = bgr_image.shape[:2]

        # --- Working Resolution Scaling for High-Res Inputs ---
        is_high_res = max(orig_h, orig_w) > 1024
        if is_high_res:
            work_scale = 768.0 / float(max(orig_h, orig_w))
            work_w = max(1, int(round(orig_w * work_scale)))
            work_h = max(1, int(round(orig_h * work_scale)))
            proc_bgr = cv2.resize(bgr_image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            proc_bgr = bgr_image
            work_w, work_h = orig_w, orig_h

        rgb_image = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

        # --- Adaptive Sensor Noise Pre-Conditioning ---
        green = rgb_image[:, :, 1].astype(np.float32)
        fov_raw = self._safe_fov_mask(proc_bgr) > 0
        if np.any(fov_raw):
            residual = np.abs(green - cv2.GaussianBlur(green, (3, 3), 0.8))
            mad_sigma = float(np.median(residual[fov_raw])) / 0.6745
            if mad_sigma > 4.2:
                h_val = float(min(11.0, max(5.0, mad_sigma * 0.65)))
                denoised = cv2.fastNlMeansDenoisingColored(
                    rgb_image, None, h=h_val, hColor=h_val, templateWindowSize=5, searchWindowSize=15
                )
                denoised[~fov_raw] = 0
                rgb_image = denoised

        binary_mask, prob_map = predict_vessel_mask(
            self.model,
            rgb_image,
            device=self.device,
            threshold=threshold,
            use_tta=use_tta,
        )

        # --- Quantile-Calibrated Adaptive Probability Thresholding ---
        if np.any(fov_raw):
            p95 = float(np.percentile(prob_map[fov_raw], 95.0))
            if p95 < 0.85:
                calibrated_at_default = max(0.32, 0.65 * p95)
                eff_thresh = max(0.10, min(threshold, (threshold / 0.60) * calibrated_at_default))
                binary_mask = (prob_map >= eff_thresh).astype(np.uint8)

        if apply_postprocess:
            binary_mask = refine_vessel_mask(
                binary_mask, min_size=min_size, max_hole_size=max_hole_size
            )

        # --- FOV Edge Artifact Suppression ---
        fov_mask = self._safe_fov_mask(proc_bgr) > 0
        fov_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fov_eroded = cv2.erode(fov_mask.astype(np.uint8), fov_kernel) > 0
        binary_mask = binary_mask & fov_eroded

        vessel_density = float(np.mean(binary_mask[fov_mask] > 0)) if np.any(fov_mask) else float(np.mean(binary_mask > 0))

        keypoints_list: list[dict[str, float]] = []
        if compute_skeleton:
            from matching.skeleton import skeletonize_zhang_suen
            from classical_pipeline.skeleton_metrics import prune_short_spurs
            from matching.bifurcation import detect_bifurcations

            skel = skeletonize_zhang_suen(binary_mask) > 0
            skel = prune_short_spurs(skel, threshold=8) & fov_eroded
            bif_pts = detect_bifurcations(skel)
            scale_x = float(orig_w) / float(work_w)
            scale_y = float(orig_h) / float(work_h)
            keypoints_list = [{"x": float(p[0] * scale_x), "y": float(p[1] * scale_y)} for p in bif_pts]
            skeleton_uint8 = (skel * 255).astype(np.uint8)
        else:
            skeleton_uint8 = np.zeros_like(binary_mask, dtype=np.uint8)

        vessel_mask_uint8 = (binary_mask * 255).astype(np.uint8)
        prob_map_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)

        if is_high_res:
            vessel_mask_uint8 = cv2.resize(vessel_mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            prob_map_uint8 = cv2.resize(prob_map_uint8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            skeleton_uint8 = cv2.resize(skeleton_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return vessel_mask_uint8, prob_map_uint8, skeleton_uint8, vessel_density, keypoints_list, elapsed_ms

    def localize_landmarks(
        self,
        bgr_image: np.ndarray,
        vessel_mask: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, float], float, float]:
        t0 = time.perf_counter()
        if vessel_mask is None or not np.any(vessel_mask > 0) or np.all(vessel_mask > 0):
            vessel_mask, _, _, _, _, _ = self.segment_vessels(bgr_image, threshold=0.30, compute_skeleton=False)

        h, w = bgr_image.shape[:2]
        fov_mask = self._safe_fov_mask(bgr_image)

        try:
            od_x, od_y = find_optic_disk(vessel_mask, bgr_image, color_order="BGR")
        except Exception:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            gray_fov = np.where(fov_mask, gray, 0)
            if np.any(gray_fov > 0):
                _, _, _, max_loc = cv2.minMaxLoc(gray_fov)
                od_x, od_y = float(max_loc[0]), float(max_loc[1])
            else:
                od_x, od_y = float(w * 0.35), float(h * 0.50)

        try:
            fovea_x, fovea_y = find_fovea(vessel_mask, (od_x, od_y), bgr_image, color_order="BGR")
        except Exception:
            sign = 1.0 if od_x < w / 2.0 else -1.0
            fovea_x = float(np.clip(od_x + sign * 0.303405 * w, 0.05 * w, 0.95 * w))
            fovea_y = float(np.clip(od_y + 0.054073 * h, 0.05 * h, 0.95 * h))

        dist = float(math.hypot(od_x - fovea_x, od_y - fovea_y))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return (
            {"x": float(od_x), "y": float(od_y)},
            {"x": float(fovea_x), "y": float(fovea_y)},
            dist,
            elapsed_ms,
        )

    @staticmethod
    def normalize_to_canonical_fov(
        bgr_image: np.ndarray, target_radius_frac: float = 0.475
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        h, w = bgr_image.shape[:2]
        min_dim = min(h, w)
        try:
            fov_mask = PipelineService._safe_fov_mask(bgr_image)
            y_indices, x_indices = np.where(fov_mask)
        except Exception:
            fov_mask = np.ones((h, w), dtype=bool)
            y_indices, x_indices = np.where(fov_mask)

        if len(x_indices) == 0:
            return bgr_image, np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), float(min(h, w) * target_radius_frac)

        cx = float(np.median(x_indices))
        cy = float(np.median(y_indices))
        span_x = float(np.percentile(x_indices, 99.5) - np.percentile(x_indices, 0.5))
        span_y = float(np.percentile(y_indices, 99.5) - np.percentile(y_indices, 0.5))
        current_radius = 0.5 * max(span_x, span_y)

        if min_dim < 450 or min_dim > 650:
            grid_scale = 565.0 / float(min_dim)
            out_w = max(1, int(round(w * grid_scale)))
            out_h = max(1, int(round(h * grid_scale)))
        else:
            grid_scale = 1.0
            out_w, out_h = w, h

        target_radius = target_radius_frac * min(out_h, out_w)
        scale = target_radius / max(current_radius, 10.0)

        if abs(scale - 1.0) < 0.03 and grid_scale == 1.0:
            return bgr_image, np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), current_radius

        M_2x3 = np.array(
            [
                [scale, 0.0, (cx * grid_scale) - scale * cx],
                [0.0, scale, (cy * grid_scale) - scale * cy],
            ],
            dtype=np.float32,
        )
        M_inv_2x3 = cv2.invertAffineTransform(M_2x3)

        normalized_bgr = cv2.warpAffine(
            bgr_image,
            M_2x3,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        M_3x3 = np.vstack([M_2x3, [0, 0, 1]]).astype(np.float32)
        M_inv_3x3 = np.vstack([M_inv_2x3, [0, 0, 1]]).astype(np.float32)

        return normalized_bgr, M_3x3, M_inv_3x3, target_radius

    def extract_full_template(
        self,
        bgr_image: np.ndarray,
        threshold: float = 0.60,
        normalize_fov: bool = True,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if normalize_fov:
            proc_img, _, M_inv, fov_rad = self.normalize_to_canonical_fov(bgr_image)
        else:
            proc_img = bgr_image
            M_inv = np.eye(3, dtype=np.float32)
            fov_rad = float(min(bgr_image.shape[:2]) * 0.475)

        vessel_mask, prob_map, skeleton, _, _, _ = self.segment_vessels(proc_img, threshold=threshold)
        od_dict, fovea_dict, _, _ = self.localize_landmarks(proc_img, vessel_mask)
        od_tuple = (od_dict["x"], od_dict["y"])
        fovea_tuple = (fovea_dict["x"], fovea_dict["y"])

        template = extract_identity_template(
            vessel_mask=vessel_mask,
            od_center=od_tuple,
            fovea_center=fovea_tuple,
            use_od_filtering=True,
            prune_spurs=True,
            skeleton=(skeleton > 0),
        )
        return template, vessel_mask, prob_map, skeleton, M_inv

    def match_identities(
        self,
        bgr_image_a: np.ndarray,
        bgr_image_b: np.ndarray,
        ratio_tolerance: float = 0.05,
        angle_tolerance_rad: float = 0.08,
        rotate_deg_b: float = 0.0,
        scale_factor_b: float = 1.0,
        brightness_b: float = 1.0,
        blur_sigma_b: float = 0.0,
        haze_factor_b: float = 0.0,
        glare_intensity_b: float = 0.0,
        noise_sigma_b: float = 0.0,
        occlusion_frac_b: float = 0.0,
        gamma_b: float = 1.0,
        spherical_warp_b: float = 0.0,
        elastic_alpha_b: float = 0.0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        # --- Image Perturbation Application ---
        has_perturbation = (
            abs(rotate_deg_b) > 1e-3
            or abs(scale_factor_b - 1.0) > 1e-3
            or abs(brightness_b - 1.0) > 1e-3
            or blur_sigma_b > 1e-3
            or haze_factor_b > 1e-3
            or glare_intensity_b > 1e-3
            or noise_sigma_b > 1e-3
            or occlusion_frac_b > 1e-3
            or abs(gamma_b - 1.0) > 1e-3
            or abs(spherical_warp_b) > 1e-3
            or elastic_alpha_b > 1e-3
        )
        if has_perturbation:
            bgr_image_b = self.apply_perturbation(
                bgr_image_b,
                rotate_deg=rotate_deg_b,
                scale_factor=scale_factor_b,
                brightness=brightness_b,
                blur_sigma=blur_sigma_b,
                haze_factor=haze_factor_b,
                glare_intensity=glare_intensity_b,
                noise_sigma=noise_sigma_b,
                occlusion_frac=occlusion_frac_b,
                gamma=gamma_b,
                spherical_warp=spherical_warp_b,
                elastic_alpha=elastic_alpha_b,
            )

        template_a, vessel_a, prob_a, skel_a, M_inv_a = self.extract_full_template(bgr_image_a)
        template_b, vessel_b, prob_b, skel_b, M_inv_b = self.extract_full_template(bgr_image_b)

        match_details = match_templates_detailed(
            template_a,
            template_b,
            ratio_tolerance=ratio_tolerance,
            angle_tolerance_rad=angle_tolerance_rad,
        )
        score = match_details["score"]
        is_match = verify_identity(score, threshold=0.20)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        def serialize_template(
            tmpl: dict[str, Any],
            v_mask: np.ndarray,
            p_map: np.ndarray,
            s_mask: np.ndarray,
            M_inv: np.ndarray,
            orig_shape: tuple[int, int],
        ) -> dict[str, Any]:
            orig_h, orig_w = orig_shape
            is_transformed = not np.allclose(M_inv[:2], np.eye(2, 3), atol=1e-4)

            def map_pt(x: float, y: float) -> tuple[float, float]:
                if not is_transformed:
                    return x, y
                mapped = M_inv[:2, :2] @ np.array([x, y], dtype=np.float64) + M_inv[:2, 2]
                return float(mapped[0]), float(mapped[1])

            keypoints = []
            for pt in tmpl["keypoints"]:
                mx, my = map_pt(pt[0], pt[1])
                keypoints.append({"x": mx, "y": my})

            triangles = []
            for inv in tmpl.get("invariants", []):
                t_idx = int(inv.get("triangle_index", inv.get("triangle_idx", 0)))
                verts = inv.get("vertices", [])
                mapped_verts = [list(map_pt(vx, vy)) for vx, vy in verts]
                mcx, mcy = map_pt(inv["centroid"][0], inv["centroid"][1])
                triangles.append({
                    "triangle_index": t_idx,
                    "vertices": mapped_verts,
                    "centroid": [mcx, mcy],
                    "adjacent_ratios": [float(r) for r in inv["adjacent_ratios"]],
                    "angles": [float(a) for a in inv.get("angles", [])],
                    "area": float(inv["area"]),
                })

            od = None
            if tmpl.get("od_center"):
                od_x, od_y = map_pt(tmpl["od_center"][0], tmpl["od_center"][1])
                od = {"x": od_x, "y": od_y}

            fov = None
            if tmpl.get("fovea_center"):
                fov_x, fov_y = map_pt(tmpl["fovea_center"][0], tmpl["fovea_center"][1])
                fov = {"x": fov_x, "y": fov_y}

            if is_transformed:
                v_disp = cv2.warpAffine(v_mask, M_inv[:2], (orig_w, orig_h), flags=cv2.INTER_NEAREST)
                p_disp = cv2.warpAffine(p_map, M_inv[:2], (orig_w, orig_h), flags=cv2.INTER_LINEAR)
                s_disp = cv2.warpAffine(s_mask, M_inv[:2], (orig_w, orig_h), flags=cv2.INTER_NEAREST)
            else:
                v_disp, p_disp, s_disp = v_mask, p_map, s_mask

            return {
                "num_keypoints": tmpl["num_keypoints"],
                "num_triangles": tmpl["delaunay_triangles"],
                "keypoints": keypoints,
                "triangles": triangles,
                "od_center": od,
                "fovea_center": fov,
                "vessel_mask_base64": self.encode_base64_image(v_disp),
                "prob_map_base64": self.encode_base64_image(p_disp),
                "skeleton_base64": self.encode_base64_image(s_disp),
                "width": orig_w,
                "height": orig_h,
            }

        clean_pairs = []
        for pair in match_details["matched_pairs"]:
            p = dict(pair)
            if not np.allclose(M_inv_a[:2], np.eye(2, 3), atol=1e-4):
                ca = M_inv_a[:2, :2] @ np.array(p["centroid_a"], dtype=np.float64) + M_inv_a[:2, 2]
                p["centroid_a"] = [float(ca[0]), float(ca[1])]
                p["vertices_a"] = [
                    [float(pt[0]), float(pt[1])]
                    for pt in (M_inv_a[:2, :2] @ np.array(p["vertices_a"]).T + M_inv_a[:2, 2:3]).T
                ]
            if not np.allclose(M_inv_b[:2], np.eye(2, 3), atol=1e-4):
                cb = M_inv_b[:2, :2] @ np.array(p["centroid_b"], dtype=np.float64) + M_inv_b[:2, 2]
                p["centroid_b"] = [float(cb[0]), float(cb[1])]
                p["vertices_b"] = [
                    [float(pt[0]), float(pt[1])]
                    for pt in (M_inv_b[:2, :2] @ np.array(p["vertices_b"]).T + M_inv_b[:2, 2:3]).T
                ]
            clean_pairs.append(p)

        return {
            "similarity_score": float(score),
            "is_match": bool(is_match),
            "threshold": 0.20,
            "matched_pairs": clean_pairs,
            "template_a": serialize_template(template_a, vessel_a, prob_a, skel_a, M_inv_a, bgr_image_a.shape[:2]),
            "template_b": serialize_template(template_b, vessel_b, prob_b, skel_b, M_inv_b, bgr_image_b.shape[:2]),
            "perturbed_image_b_base64": self.encode_base64_image(bgr_image_b, ".png"),
            "elapsed_ms": elapsed_ms,
        }
