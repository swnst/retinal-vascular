import {
  LocalizationResult,
  MatchResult,
  NormalizedImageResult,
  PerturbationParams,
  PresetItem,
  SegmentationResult,
} from "@/types/biometrics";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// --- API Client Handlers ---
export async function fetchHealth(): Promise<{ status: string; device: string; model_loaded: boolean }> {
  const res = await fetch(`${API_BASE}/api/v1/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API server is offline");
  return res.json();
}

export async function fetchPresets(): Promise<PresetItem[]> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/presets`, { cache: "no-store" });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function requestSegmentation(
  imageBase64: string,
  threshold: number = 0.60,
  applyPostprocess: boolean = true,
  minSize: number = 25,
  maxHoleSize: number = 15,
  useTta: boolean = true,
): Promise<SegmentationResult> {
  const res = await fetch(`${API_BASE}/api/v1/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: imageBase64,
      threshold,
      apply_postprocess: applyPostprocess,
      min_size: minSize,
      max_hole_size: maxHoleSize,
      use_tta: useTta,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Segmentation failed" }));
    throw new Error(err.detail || "Segmentation failed");
  }
  return res.json();
}

export async function requestLocalization(
  imageBase64: string,
  vesselMaskBase64?: string
): Promise<LocalizationResult> {
  const res = await fetch(`${API_BASE}/api/v1/localize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: imageBase64,
      vessel_mask_base64: vesselMaskBase64,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Localization failed" }));
    throw new Error(err.detail || "Localization failed");
  }
  return res.json();
}

export async function requestMatch(
  imageABase64: string,
  imageBBase64: string,
  ratioTolerance: number = 0.05,
  angleToleranceRad: number = 0.08,
  rotateDegB: number = 0.0,
  scaleFactorB: number = 1.0,
  brightnessB: number = 1.0,
  additionalPerturbations?: Partial<PerturbationParams>,
): Promise<MatchResult> {
  const payload = {
    image_a_base64: imageABase64,
    image_b_base64: imageBBase64,
    ratio_tolerance: ratioTolerance,
    angle_tolerance_rad: angleToleranceRad,
    rotate_deg_b: rotateDegB,
    scale_factor_b: scaleFactorB,
    brightness_b: brightnessB,
    blur_sigma_b: additionalPerturbations?.blur_sigma_b ?? 0.0,
    haze_factor_b: additionalPerturbations?.haze_factor_b ?? 0.0,
    glare_intensity_b: additionalPerturbations?.glare_intensity_b ?? 0.0,
    noise_sigma_b: additionalPerturbations?.noise_sigma_b ?? 0.0,
    occlusion_frac_b: additionalPerturbations?.occlusion_frac_b ?? 0.0,
    gamma_b: additionalPerturbations?.gamma_b ?? 1.0,
    spherical_warp_b: additionalPerturbations?.spherical_warp_b ?? 0.0,
    elastic_alpha_b: additionalPerturbations?.elastic_alpha_b ?? 0.0,
  };

  const res = await fetch(`${API_BASE}/api/v1/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Biometric matching failed" }));
    throw new Error(err.detail || "Biometric matching failed");
  }
  return res.json();
}

export async function uploadClinicalImage(file: File, maxDim: number = 768): Promise<NormalizedImageResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("max_dim", maxDim.toString());

  const res = await fetch(`${API_BASE}/api/v1/upload-image`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Image upload failed" }));
    throw new Error(err.detail || "Image upload failed");
  }
  return res.json();
}

