from pydantic import BaseModel, Field


# --- API Request and Response Models ---
class SegmentationRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded retinal fundus image (PNG/JPEG)")
    threshold: float = Field(0.60, ge=0.0, le=1.0, description="Sigmoid vessel binarization threshold")
    apply_postprocess: bool = Field(True, description="Apply morphological noise removal & hole filling")
    min_size: int = Field(25, ge=1, description="Minimum component size in pixels")
    max_hole_size: int = Field(15, ge=0, description="Maximum hole size to fill")
    use_tta: bool = Field(True, description="Enable test-time augmentation")


class SegmentationResponse(BaseModel):
    vessel_mask_base64: str
    prob_map_base64: str
    skeleton_base64: str
    vessel_density: float
    keypoints: list[dict[str, float]] = Field(default_factory=list)
    num_keypoints: int = 0
    elapsed_ms: float
    width: int
    height: int


class LocalizationRequest(BaseModel):
    image_base64: str
    vessel_mask_base64: str | None = None


class LocalizationResponse(BaseModel):
    optic_disc: dict[str, float]
    fovea: dict[str, float]
    disc_to_fovea_distance: float
    elapsed_ms: float
    width: int
    height: int


class MatchRequest(BaseModel):
    image_a_base64: str
    image_b_base64: str
    ratio_tolerance: float = Field(0.05, ge=0.01, le=0.2)
    angle_tolerance_rad: float = Field(0.08, ge=0.01, le=0.5)
    # --- 11-Parameter Physical & Clinical Perturbation Suite ---
    rotate_deg_b: float = Field(0.0, ge=-180.0, le=180.0, description="Patient head tilt / cyclorotation in degrees")
    scale_factor_b: float = Field(1.0, ge=0.50, le=2.00, description="Camera working-distance scale factor")
    brightness_b: float = Field(1.0, ge=0.20, le=3.00, description="Flash discharge intensity / illumination factor")
    blur_sigma_b: float = Field(0.0, ge=0.0, le=10.0, description="Optical defocus / accommodation blur sigma in pixels")
    haze_factor_b: float = Field(0.0, ge=0.0, le=0.95, description="Cataract / vitreous media opacity haze factor")
    glare_intensity_b: float = Field(0.0, ge=0.0, le=1.0, description="Corneal specular reflection / flash hotspot intensity")
    noise_sigma_b: float = Field(0.0, ge=0.0, le=50.0, description="CMOS sensor thermal / shot ISO noise sigma")
    occlusion_frac_b: float = Field(0.0, ge=0.0, le=0.60, description="Eyelid closure / eyelash shadow occlusion fraction")
    gamma_b: float = Field(1.0, ge=0.20, le=3.00, description="ISP tone-mapping curve gamma exponent")
    spherical_warp_b: float = Field(0.0, ge=-0.50, le=0.50, description="Off-axis gaze spherical radial distortion coefficient")
    elastic_alpha_b: float = Field(0.0, ge=0.0, le=10.0, description="Pulsatile IOP / micro-saccadic elastic displacement amplitude")


class LandmarkPoint(BaseModel):
    x: float
    y: float


class TriangleInvariant(BaseModel):
    triangle_index: int
    vertices: list[list[float]]
    centroid: list[float]
    adjacent_ratios: list[float]
    angles: list[float]
    area: float


class MatchedTrianglePair(BaseModel):
    idx_a: int
    idx_b: int
    cost: float
    centroid_a: list[float]
    centroid_b: list[float]
    vertices_a: list[list[float]]
    vertices_b: list[list[float]]


class TemplateSummary(BaseModel):
    num_keypoints: int
    num_triangles: int
    keypoints: list[LandmarkPoint]
    triangles: list[TriangleInvariant]
    od_center: dict[str, float] | None = None
    fovea_center: dict[str, float] | None = None
    vessel_mask_base64: str
    prob_map_base64: str
    skeleton_base64: str
    width: int
    height: int


class MatchResponse(BaseModel):
    similarity_score: float
    is_match: bool
    threshold: float
    matched_pairs: list[MatchedTrianglePair]
    template_a: TemplateSummary
    template_b: TemplateSummary
    perturbed_image_b_base64: str | None = None
    elapsed_ms: float


class PresetItem(BaseModel):
    id: str
    title: str
    description: str
    dataset: str
    scenario: str
    target_stage: str = Field("ocular", description="'ocular' for Step 1, 'matching' for Step 2")
    badge_label: str = Field("", description="Concise stage badge text")
    image_base64: str
    image_b_base64: str | None = None
    thumbnail_base64: str


# --- Image Normalization Models ---
class NormalizeImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded clinical fundus image in any format")
    max_dim: int = Field(768, ge=128, le=2048, description="Maximum output dimension preserving aspect ratio")


class NormalizeImageResponse(BaseModel):
    image_base64: str
    width: int
    height: int
    original_width: int
    original_height: int
    filename: str = ""

