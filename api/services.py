import base64
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

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
    def decode_base64_image(image_base64: str) -> np.ndarray:
        if len(image_base64) > 35_000_000:
            raise ValueError("Image payload exceeds maximum allowed size.")
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        raw_bytes = base64.b64decode(image_base64)
        if len(raw_bytes) == 0:
            raise ValueError("Image payload is empty.")
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode base64 into a valid image.")
        if img.shape[0] > 4096 or img.shape[1] > 4096:
            raise ValueError(f"Image dimensions ({img.shape[1]}x{img.shape[0]}) exceed maximum allowed 4096x4096.")
        return img

    @staticmethod
    def encode_base64_image(image_np: np.ndarray, format_ext: str = ".png") -> str:
        success, buffer = cv2.imencode(format_ext, image_np)
        if not success:
            raise ValueError("Failed to encode image to buffer.")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/png" if format_ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64_str}"

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
        fov = estimate_fov_mask(img_bgr) > 0

        # 1. Gamma Tone-Curve Adjustment
        if abs(gamma - 1.0) > 1e-3:
            out = np.power(np.clip(out / 255.0, 0.0, 1.0), gamma) * 255.0

        # 2. Cataract / Media Opacity Haze
        if haze_factor > 1e-3:
            veil = np.zeros_like(out)
            veil[:, :, 0] = 125.0
            veil[:, :, 1] = 165.0
            veil[:, :, 2] = 210.0
            blurred_bg = cv2.GaussianBlur(out, (0, 0), sigmaX=14.0)
            haze_layer = 0.65 * veil + 0.35 * blurred_bg
            out = (1.0 - haze_factor) * out + haze_factor * haze_layer

        # 3. Corneal Specular Flash Glare
        if glare_intensity > 1e-3:
            y_grid, x_grid = np.indices((h, w), dtype=np.float32)
            gx, gy = w * 0.45, h * 0.42
            sigma_g = min(h, w) * 0.14
            spot = np.exp(-0.5 * (((x_grid - gx) / sigma_g) ** 2 + ((y_grid - gy) / sigma_g) ** 2))
            glare_bgr = np.stack([spot * 220.0, spot * 240.0, spot * 255.0], axis=-1)
            out = out + glare_intensity * glare_bgr

        # 4. Defocus / Accommodation Blur
        if blur_sigma > 1e-3:
            ksize = int(2 * round(3 * blur_sigma) + 1)
            ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
            out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

        # 5. CMOS Sensor Thermal/ISO Noise
        if noise_sigma > 1e-3:
            rng = np.random.default_rng(42)
            gaussian_noise = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
            out = out + gaussian_noise

        # 6. Flash Illumination / Brightness
        if abs(brightness - 1.0) > 1e-3:
            out = out * brightness

        # 7. Eyelid Closure / Eyelash Shadow Occlusion
        if occlusion_frac > 1e-3:
            ys, _ = np.where(fov)
            if len(ys) > 0:
                y_min, y_max = int(np.min(ys)), int(np.max(ys))
                cut_top = y_min + int((y_max - y_min) * (occlusion_frac * 0.65))
                cut_bot = y_max - int((y_max - y_min) * (occlusion_frac * 0.35))
                out[:cut_top, :] = 0
                out[cut_bot:, :] = 0

        # Mask outside clean FOV before geometric warps
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        out[~fov] = 0

        # 8. Spherical Radial Distortion
        if abs(spherical_warp) > 1e-3:
            cx, cy = w / 2.0, h / 2.0
            r_max = math.hypot(cx, cy)
            y_grid, x_grid = np.indices((h, w), dtype=np.float32)
            dx = x_grid - cx
            dy = y_grid - cy
            r = np.sqrt(dx * dx + dy * dy)
            r_norm = r / max(r_max, 1.0)
            distortion_factor = 1.0 + spherical_warp * (r_norm ** 2)
            map_x = (cx + dx * distortion_factor).astype(np.float32)
            map_y = (cy + dy * distortion_factor).astype(np.float32)
            out = cv2.remap(out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 9. Elastic Displacement Field (Pulsatile IOP / Micro-saccades)
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

        # 10 & 11. Rigid Affine Rotation and Scaling
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
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Adaptive high-frequency sensor noise pre-conditioning (Donoho-Johnstone MAD estimator)
        green = rgb_image[:, :, 1].astype(np.float32)
        fov_raw = estimate_fov_mask(bgr_image) > 0
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

        # Quantile-Calibrated Adaptive Probability Thresholding
        if np.any(fov_raw) and abs(threshold - 0.60) < 1e-3:
            p95 = float(np.percentile(prob_map[fov_raw], 95.0))
            if p95 < 0.85:
                eff_thresh = max(0.32, min(threshold, 0.65 * p95))
                binary_mask = (prob_map >= eff_thresh).astype(np.uint8)

        if apply_postprocess:
            binary_mask = refine_vessel_mask(
                binary_mask, min_size=min_size, max_hole_size=max_hole_size
            )

        # FOV masking & suppression of circular edge artifacts
        fov_mask = estimate_fov_mask(bgr_image) > 0
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
            keypoints_list = [{"x": float(p[0]), "y": float(p[1])} for p in bif_pts]
            skeleton_uint8 = (skel * 255).astype(np.uint8)
        else:
            skeleton_uint8 = np.zeros_like(binary_mask, dtype=np.uint8)

        vessel_mask_uint8 = (binary_mask * 255).astype(np.uint8)
        prob_map_uint8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return vessel_mask_uint8, prob_map_uint8, skeleton_uint8, vessel_density, keypoints_list, elapsed_ms

    def localize_landmarks(
        self,
        bgr_image: np.ndarray,
        vessel_mask: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, float], float, float]:
        t0 = time.perf_counter()
        if vessel_mask is None:
            vessel_mask, _, _, _, _, _ = self.segment_vessels(bgr_image, compute_skeleton=False)

        od_x, od_y = find_optic_disk(vessel_mask, bgr_image, color_order="BGR")
        fovea_x, fovea_y = find_fovea(vessel_mask, (od_x, od_y), bgr_image, color_order="BGR")

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
        try:
            fov_mask = estimate_fov_mask(bgr_image)
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

        target_radius = target_radius_frac * min(h, w)
        scale = target_radius / max(current_radius, 10.0)

        if abs(scale - 1.0) < 0.03:
            return bgr_image, np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), current_radius

        M_2x3 = cv2.getRotationMatrix2D((cx, cy), 0.0, scale)
        M_inv_2x3 = cv2.invertAffineTransform(M_2x3)

        normalized_bgr = cv2.warpAffine(
            bgr_image,
            M_2x3,
            (w, h),
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

        # Apply 11-parameter perturbation suite to Image B if any parameter is active
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
