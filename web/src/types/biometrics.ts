// --- VERA Type Declarations ---

export interface PresetItem {
  id: string;
  title: string;
  description: string;
  dataset: string;
  scenario: string;
  target_stage: "ocular" | "matching";
  badge_label: string;
  image_base64: string;
  image_b_base64?: string | null;
  thumbnail_base64: string;
}

export interface PerturbationParams {
  rotate_deg_b: number;
  scale_factor_b: number;
  brightness_b: number;
  blur_sigma_b: number;
  haze_factor_b: number;
  glare_intensity_b: number;
  noise_sigma_b: number;
  occlusion_frac_b: number;
  gamma_b: number;
  spherical_warp_b: number;
  elastic_alpha_b: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
}

export interface TriangleInvariant {
  triangle_index: number;
  vertices: [number, number][];
  centroid: [number, number];
  adjacent_ratios: number[];
  angles: number[];
  area: number;
}

export interface MatchedTrianglePair {
  idx_a: number;
  idx_b: number;
  cost: number;
  centroid_a: [number, number];
  centroid_b: [number, number];
  vertices_a: [number, number][];
  vertices_b: [number, number][];
}

export interface TemplateSummary {
  num_keypoints: number;
  num_triangles: number;
  keypoints: LandmarkPoint[];
  triangles: TriangleInvariant[];
  od_center: { x: number; y: number } | null;
  fovea_center: { x: number; y: number } | null;
  vessel_mask_base64: string;
  prob_map_base64: string;
  skeleton_base64: string;
  width: number;
  height: number;
}

export interface SegmentationResult {
  vessel_mask_base64: string;
  prob_map_base64: string;
  skeleton_base64: string;
  vessel_density: number;
  keypoints?: LandmarkPoint[];
  num_keypoints?: number;
  elapsed_ms: number;
  width: number;
  height: number;
}

export interface LocalizationResult {
  optic_disc: { x: number; y: number };
  fovea: { x: number; y: number };
  disc_to_fovea_distance: number;
  elapsed_ms: number;
  width: number;
  height: number;
}

export interface MatchResult {
  similarity_score: number;
  is_match: boolean;
  threshold: number;
  matched_pairs: MatchedTrianglePair[];
  template_a: TemplateSummary;
  template_b: TemplateSummary;
  perturbed_image_b_base64?: string;
  elapsed_ms: number;
}

export interface NormalizedImageResult {
  image_base64: string;
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  filename: string;
}

