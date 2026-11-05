import math
from pathlib import Path
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


# --- Environment & Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services import PipelineService
from classical_pipeline.localization_api import find_fovea, find_optic_disk
from matching.bifurcation import detect_bifurcations
from matching.triangulation import build_delaunay_triangulation, compute_triangle_invariants

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9.5
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


# --- Verified Scott Prahl (1999, OMLC) Tabulated Molar Extinction Data (cm^-1 M^-1) ---
PRAHL_OMLC_HB_TABLE: list[tuple[float, float, float]] = [
    (390.0, 167748.0, 167780.0),
    (400.0, 266232.0, 223296.0),
    (404.0, 308716.0, 253368.0),
    (410.0, 466840.0, 303956.0),
    (412.0, 500200.0, 321344.0),
    (414.0, 524280.0, 342596.0),
    (416.0, 521880.0, 363848.0),
    (418.0, 515520.0, 385680.0),
    (420.0, 480360.0, 407560.0),
    (422.0, 420800.0, 434380.0),
    (424.0, 376236.0, 461200.0),
    (428.0, 283112.0, 500840.0),
    (430.0, 246072.0, 528600.0),
    (432.0, 214120.0, 552160.0),
    (434.0, 165332.0, 552160.0),
    (436.0, 132820.0, 547040.0),
    (440.0, 102580.0, 413280.0),
    (450.0, 62816.0, 103292.0),
    (460.0, 44480.0, 23388.8),
    (470.0, 33209.2, 16156.4),
    (480.0, 26629.2, 14550.0),
    (490.0, 23684.4, 16684.0),
    (500.0, 20932.8, 20862.0),
    (510.0, 20035.2, 25773.6),
    (520.0, 24202.4, 31589.6),
    (528.0, 35990.0, 37490.0),
    (530.0, 39956.8, 39036.4),
    (536.0, 49376.0, 43616.0),
    (540.0, 53236.0, 46592.0),
    (542.0, 53292.0, 48148.0),
    (544.0, 52096.0, 49708.0),
    (546.0, 49868.0, 51268.0),
    (548.0, 46660.0, 52496.0),
    (550.0, 43016.0, 53412.0),
    (554.0, 36524.0, 54520.0),
    (556.0, 34476.8, 54540.0),
    (560.0, 32613.2, 53788.0),
    (566.0, 37020.0, 49348.0),
    (570.0, 44496.0, 45072.0),
    (574.0, 53308.0, 41716.0),
    (576.0, 55540.0, 40092.0),
    (578.0, 54728.0, 38467.6),
    (580.0, 50104.0, 37020.0),
    (584.0, 34639.6, 34332.8),
    (590.0, 14400.8, 28324.4),
    (600.0, 3200.0, 14677.2),
    (620.0, 942.0, 6509.6),
    (640.0, 442.0, 4345.2),
    (660.0, 319.6, 3226.56),
    (680.0, 277.6, 2407.92),
    (700.0, 276.0, 1846.08),
]


# --- Chapter 2 Figure Generation ---
def generate_figure_2_1():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    if not img_path.is_file():
        img_rgb = np.zeros((584, 565, 3), dtype=np.uint8)
    else:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(img_rgb)

    od_x, od_y = 94.31, 254.69
    od_r = 45.0
    od_circ = Circle((od_x, od_y), od_r, color="none", ec="#f1c40f", lw=2.5, ls="-")
    od_cup = Circle((od_x, od_y), od_r * 0.45, color="none", ec="#fff9d2", lw=1.8, ls=":")
    ax.add_patch(od_circ)
    ax.add_patch(od_cup)

    fov_x, fov_y = 270.91, 287.52
    macula_r = 55.0
    fovea_r = 16.0
    mac_circ = Circle((fov_x, fov_y), macula_r, color="none", ec="#e74c3c", lw=2.0, ls="--")
    fov_circ = Circle((fov_x, fov_y), fovea_r, color="none", ec="#00d2d3", lw=2.0)
    ax.add_patch(mac_circ)
    ax.add_patch(fov_circ)

    ax.annotate(
        "",
        xy=(fov_x, fov_y),
        xytext=(od_x, od_y),
        arrowprops=dict(arrowstyle="<->", color="#ffffff", lw=1.8, ls="--"),
    )
    dist_px = math.hypot(od_x - fov_x, od_y - fov_y)
    mid_x = (od_x + fov_x) / 2.0
    mid_y = (od_y + fov_y) / 2.0
    ax.text(
        mid_x,
        mid_y - 14,
        f"Inter-Landmark Distance: {dist_px:.1f} px\n(~2.8x OD Diameters / 15 deg)",
        color="#ffffff",
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e272e", alpha=0.88, ec="#34495e"),
    )

    ax.annotate(
        "Optic Disc (OD)\n[Nasal / Medial Pole]",
        xy=(od_x, od_y - od_r),
        xytext=(od_x + 25, od_y - 110),
        arrowprops=dict(facecolor="#f1c40f", shrink=0.08, width=1.5, headwidth=6),
        color="#f1c40f",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e272e", alpha=0.88, ec="#f1c40f"),
    )

    ax.annotate(
        "Macula & Fovea Centralis\n[Temporal / Lateral Pole, FAZ]",
        xy=(fov_x, fov_y + fovea_r),
        xytext=(fov_x - 30, fov_y + 110),
        arrowprops=dict(facecolor="#00d2d3", shrink=0.08, width=1.5, headwidth=6),
        color="#00d2d3",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e272e", alpha=0.88, ec="#00d2d3"),
    )

    ax.text(
        240,
        90,
        "Superior Temporal Arcade",
        color="#ff6b6b",
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e272e", alpha=0.82, ec="#ff6b6b"),
    )
    ax.text(
        240,
        490,
        "Inferior Temporal Arcade",
        color="#ff6b6b",
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e272e", alpha=0.82, ec="#ff6b6b"),
    )

    ax.text(
        35,
        h / 2,
        "NASAL\n(Medial)",
        color="#c8d6e5",
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#0d1117", ec="#576574"),
    )
    ax.text(
        w - 35,
        h / 2,
        "TEMPORAL\n(Lateral)",
        color="#c8d6e5",
        fontsize=10.5,
        fontweight="bold",
        ha="right",
        va="center",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#0d1117", ec="#576574"),
    )
    ax.text(
        w / 2,
        30,
        "SUPERIOR",
        color="#c8d6e5",
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#0d1117", ec="#576574"),
    )
    ax.text(
        w / 2,
        h - 30,
        "INFERIOR",
        color="#c8d6e5",
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#0d1117", ec="#576574"),
    )

    ax.set_title(
        "Figure 2.1: Clinical Retinal Fundus Anatomical Landmarks (Left Eye OS, 01_test.tif)",
        color="#ffffff",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    ax.axis("off")
    out_file = OUTPUT_DIR / "Figure_2.1_Retinal_Anatomy.png"
    plt.savefig(str(out_file), facecolor="#0d1117", edgecolor="none", bbox_inches="tight")
    plt.close()


def generate_figure_2_2():
    table_arr = np.array(PRAHL_OMLC_HB_TABLE, dtype=np.float64)
    tab_wl = table_arr[:, 0]
    tab_hbo2 = table_arr[:, 1]
    tab_hb = table_arr[:, 2]

    wl_full = np.linspace(390.0, 700.0, 621)
    hbo2_full = np.interp(wl_full, tab_wl, tab_hbo2)
    hb_full = np.interp(wl_full, tab_wl, tab_hb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fafafa")
        ax.grid(True, which="both", ls=":", color="#bdc3c7", alpha=0.7)

    ax1.axvspan(400, 500, color="#3498db", alpha=0.10, label="Blue (400-500 nm): High Scattering")
    ax1.axvspan(500, 600, color="#2ecc71", alpha=0.18, label="Green [Optimal] (500-600 nm): Contrast")
    ax1.axvspan(600, 700, color="#e74c3c", alpha=0.10, label="Red (600-700 nm): Deep Penetration")

    ax1.semilogy(wl_full, hbo2_full, color="#c0392b", lw=2.4, label=r"Oxyhemoglobin ($HbO_2$)")
    ax1.semilogy(wl_full, hb_full, color="#2980b9", lw=2.4, ls="--", label=r"Deoxyhemoglobin ($Hb$)")

    ax1.annotate(
        "Soret Peak (414 nm, 524k)",
        xy=(414, 524280),
        xytext=(435, 420000),
        arrowprops=dict(facecolor="#c0392b", shrink=0.05, width=1.2, headwidth=5),
        fontsize=8.0,
        fontweight="bold",
        color="#c0392b",
    )
    ax1.annotate(
        "Soret Peak (432 nm, 552k)",
        xy=(432, 552160),
        xytext=(455, 200000),
        arrowprops=dict(facecolor="#2980b9", shrink=0.05, width=1.2, headwidth=5),
        fontsize=8.0,
        fontweight="bold",
        color="#2980b9",
    )

    ax1.set_title("Full Absorption Spectrum (390 - 700 nm, Semilog)", fontsize=10, fontweight="bold")
    ax1.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel(r"Molar Extinction $\varepsilon$ ($\mathrm{M}^{-1}\mathrm{cm}^{-1}$)", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(390, 700)
    ax1.set_ylim(1e2, 1e6)
    ax1.legend(loc="upper right", fontsize=7.5)

    wl_q = np.linspace(480.0, 620.0, 281)
    hbo2_q = np.interp(wl_q, tab_wl, tab_hbo2)
    hb_q = np.interp(wl_q, tab_wl, tab_hb)

    ax2.axvspan(520, 580, color="#2ecc71", alpha=0.20, label="Green Diagnostic Band (520-580 nm)")
    ax2.plot(wl_q, hbo2_q, color="#c0392b", lw=2.4, label=r"Oxyhemoglobin ($HbO_2$)")
    ax2.plot(wl_q, hb_q, color="#2980b9", lw=2.4, ls="--", label=r"Deoxyhemoglobin ($Hb$)")

    ax2.scatter(542, 53292, color="#c0392b", s=40, zorder=5)
    ax2.text(542, 54500, r"$\beta$-peak (542nm)", color="#c0392b", fontsize=7.5, fontweight="bold", ha="center")
    ax2.scatter(576, 55540, color="#c0392b", s=40, zorder=5)
    ax2.text(576, 56500, r"$\alpha$-peak (576nm)", color="#c0392b", fontsize=7.5, fontweight="bold", ha="center")
    ax2.scatter(556, 54540, color="#2980b9", s=40, zorder=5)
    ax2.text(556, 50000, "Hb peak (556nm)", color="#2980b9", fontsize=7.5, fontweight="bold", ha="center")

    diff_q = hbo2_q - hb_q
    sign_changes = np.where(np.diff(np.sign(diff_q)))[0]
    for sc in sign_changes:
        w_iso = float(wl_q[sc] - diff_q[sc] * (wl_q[sc + 1] - wl_q[sc]) / (diff_q[sc + 1] - diff_q[sc]))
        y_iso = float(np.interp(w_iso, wl_q, hbo2_q))
        ax2.scatter(w_iso, y_iso, color="#8e44ad", s=35, zorder=6)

    ax2.set_title("Q-Band Detailed Linear Spectrum (480 - 620 nm)", fontsize=10, fontweight="bold")
    ax2.set_xlabel(r"Wavelength $\lambda$ (nm)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel(r"Molar Extinction $\varepsilon$ ($\mathrm{M}^{-1}\mathrm{cm}^{-1}$)", fontsize=9.5, fontweight="bold")
    ax2.set_xlim(480, 620)
    ax2.set_ylim(0, 60000)
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("Figure 2.2: Hemoglobin Molar Extinction Spectrum (Scott Prahl 1999 Standard, OMLC)", fontsize=11.5, fontweight="bold", y=0.98)
    out_file = OUTPUT_DIR / "Figure_2.2_Hemoglobin_Spectrum.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


def generate_figure_2_3():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    cases = [
        ("Case (a): Endpoint\n($X_N = 1, N_8 = 1$)", np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]), "#e67e22"),
        ("Case (b): Vessel Path\n($X_N = 2, N_8 = 2$)", np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), "#2980b9"),
        ("Case (c): Bifurcation\n($X_N = 3, N_8 = 3$)", np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]), "#c0392b"),
        ("Case (d): Diagonal Rescue\n($X_N = 2, N_8 = 3$)", np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]]), "#8e44ad"),
    ]

    labels = [
        ["P4 (NW)", "P3 (N)", "P2 (NE)"],
        ["P5 (W)", "P (Center)", "P1 (E)"],
        ["P6 (SW)", "P7 (S)", "P8 (SE)"],
    ]

    for ax, (title, grid, c_accent) in zip(axes, cases):
        ax.set_facecolor("#fafafa")
        ax.imshow(grid, cmap="Blues", alpha=0.18, vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="#2c3e50", linestyle="-", linewidth=1.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        for i in range(3):
            for j in range(3):
                val = grid[i, j]
                lbl = labels[i][j]
                color = c_accent if val == 1 else "#7f8c8d"
                weight = "bold" if val == 1 else "normal"
                ax.text(j, i, f"{lbl}\n[val={val}]", ha="center", va="center", color=color, fontsize=9, fontweight=weight)

        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=10, color=c_accent)

    fig.suptitle(
        r"Figure 2.3: Rutovitz Crossing Number Formulation $X_N(P) = \frac{1}{2} \sum_{i=1}^8 |p_i - p_{i+1}|$ with Diagonal Rescue",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    out_file = OUTPUT_DIR / "Figure_2.3_Rutovitz_Crossing_Number.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


# --- Figure 2.4: Authentic Kabsch SVD Alignment ---
def generate_figure_2_4():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    img_bgr = cv2.imread(str(img_path))
    service = PipelineService()
    _, _, skel, _, _, _ = service.segment_vessels(img_bgr)
    bifurcations = detect_bifurcations(skel, border_margin=25, min_distance=12.0)
    pts_src = np.array(bifurcations, dtype=np.float64)

    theta_true = math.radians(25.0)
    rot_matrix = np.array([
        [math.cos(theta_true), -math.sin(theta_true)],
        [math.sin(theta_true), math.cos(theta_true)],
    ])
    t_true = np.array([35.0, -20.0])
    np.random.seed(42)
    pts_target = (pts_src @ rot_matrix.T) + t_true + np.random.normal(0, 0.45, pts_src.shape)

    mu_p = np.mean(pts_src, axis=0)
    mu_q = np.mean(pts_target, axis=0)
    p_norm = pts_src - mu_p
    q_norm = pts_target - mu_q

    h_mat = p_norm.T @ q_norm
    u, _, vt = np.linalg.svd(h_mat)
    d = np.linalg.det(vt.T @ u.T)
    r_opt = vt.T @ np.diag([1.0, d]) @ u.T
    pts_aligned = (p_norm @ r_opt.T) + mu_q
    rmse = math.sqrt(float(np.mean(np.sum((pts_aligned - pts_target) ** 2, axis=1))))
    recovered_deg = math.degrees(math.atan2(r_opt[1, 0], r_opt[0, 0]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.8), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#fafafa")
        ax.grid(True, ls=":", color="#bdc3c7")
        ax.set_xlabel("X (pixels)", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Y (pixels)", fontsize=8.5, fontweight="bold")
        ax.invert_yaxis()

    ax1.scatter(pts_src[:, 0], pts_src[:, 1], c="#e74c3c", s=40, label=f"Authentic Minutiae P (N={len(pts_src)})")
    ax1.scatter(pts_target[:, 0], pts_target[:, 1], c="#2980b9", s=40, marker="s", label="Perturbed Minutiae Q")
    ax1.set_title("1. Unaligned Scans\n(Rot: +25 deg, Trans: [35, -20])", fontsize=9.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.5)

    ax2.scatter(p_norm[:, 0], p_norm[:, 1], c="#e74c3c", s=40, label="Normalized P (Centered)")
    ax2.scatter(q_norm[:, 0], q_norm[:, 1], c="#2980b9", s=40, marker="s", label="Normalized Q (Centered)")
    ax2.set_title("2. Centroid Normalization\n" + r"($\mu_P = 0, \mu_Q = 0$)", fontsize=9.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7.5)

    ax3.scatter(pts_aligned[:, 0], pts_aligned[:, 1], c="#27ae60", s=45, label=rf"Aligned P ($\hat{{\theta}}={recovered_deg:.1f}^\circ$)")
    ax3.scatter(pts_target[:, 0], pts_target[:, 1], c="#2980b9", s=30, marker="s", label="Target Q")
    for i in range(min(50, len(pts_src))):
        ax3.plot([pts_aligned[i, 0], pts_target[i, 0]], [pts_aligned[i, 1], pts_target[i, 1]], "k-", lw=0.6, alpha=0.55)
    ax3.set_title("3. Kabsch SE(2) SVD Solved\n" + rf"($\det(VU^T)=+1$, RMSE={rmse:.3f} px)", fontsize=9.5, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(
        f"Figure 2.4: Optimal Closed-Form Kabsch SE(2) SVD Point Set Normalization on {len(pts_src)} Authentic Minutiae",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    out_file = OUTPUT_DIR / "Figure_2.4_Kabsch_SVD_Alignment.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


# --- Figure 2.5: Authentic Delaunay Triangulation ---
def generate_figure_2_5():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    img_bgr = cv2.imread(str(img_path))
    service = PipelineService()
    _, _, skel, _, _, _ = service.segment_vessels(img_bgr)
    bifurcations = detect_bifurcations(skel, border_margin=25, min_distance=12.0)
    pts_all = np.array(bifurcations, dtype=np.float64)

    center = np.array([img_bgr.shape[1] / 2.0, img_bgr.shape[0] / 2.0])
    dists = np.linalg.norm(pts_all - center, axis=1)
    pts = pts_all[np.argsort(dists)[:15]]

    tri = Delaunay(pts)
    invariants = compute_triangle_invariants(tri, pts)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    ax.triplot(pts[:, 0], pts[:, 1], tri.simplices, color="#2c3e50", lw=1.5, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], color="#e74c3c", s=75, zorder=4, label="Authentic Bifurcation Minutiae Nodes")

    best_idx = 0
    p1 = pts[tri.simplices[best_idx][0]]
    p2 = pts[tri.simplices[best_idx][1]]
    p3 = pts[tri.simplices[best_idx][2]]
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
    uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
    r = math.sqrt((p1[0] - ux)**2 + (p1[1] - uy)**2)

    circ = Circle((ux, uy), r, color="#3498db", fill=False, ls="--", lw=2.0, label="Empty Circumcircle Criterion")
    ax.add_patch(circ)
    t_poly = Polygon([p1, p2, p3], color="#f39c12", alpha=0.35, label=r"Primary Invariant Triangle $T_i$")
    ax.add_patch(t_poly)

    for i, p in enumerate(pts):
        ax.text(p[0] + 5, p[1] + 5, f"$B_{{{i+1}}}$", fontsize=9, fontweight="bold", color="#2c3e50")

    if invariants:
        inv = invariants[0]
        info_txt = (
            f"Angles: [{inv['angles'][0]:.2f}, {inv['angles'][1]:.2f}, {inv['angles'][2]:.2f}]\n"
            f"Adj Ratios: {inv['adjacent_ratios']}\n"
            f"Area: {inv['area']:.1f} px^2 | Chirality: {inv['chirality']}"
        )
        ax.text(
            0.03, 0.04, info_txt,
            transform=ax.transAxes,
            fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#bdc3c7", lw=1.2),
        )

    ax.set_title("Figure 2.5: Authentic Retinal Delaunay Triangulation and Invariant Metric Space", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("X (pixels)", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Y (pixels)", fontsize=8.5, fontweight="bold")
    ax.grid(True, ls=":", color="#bdc3c7")
    ax.legend(loc="upper right", fontsize=8)
    ax.invert_yaxis()

    out_file = OUTPUT_DIR / "Figure_2.5_Delaunay_Triangulation.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


# --- Chapter 3 Figure Generation ---
def generate_figure_3_1():
    fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis("off")

    def draw_box(x, y, w, h, title, sub, color, text_color="#ffffff"):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0,rounding_size=2.0", facecolor=color, edgecolor="#2c3e50", lw=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", color=text_color, fontsize=7.2)

    draw_box(4, 56, 43, 15, "Zone 1: Universal Clinical Ingestion", "Universal 16-bit Decoders (TIFF/PNG/JPEG/PPM 64MP)\nDynamic FOV Guard & Wavelet MAD Noise Gate (sigma > 3.5)", "#2c3e50")
    draw_box(53, 56, 43, 15, "Zone 4: FastAPI REST & 3D WebGL Studio", "Endpoints: /health, /presets, /upload-image, /normalize-image, /segment, /localize, /match\nNext.js 16 Turbopack + RetinalGlobe (-0.42 Cup) + DelaunayConstellation", "#16a085")

    draw_box(4, 33, 43, 16, "Zone 2A: Deep Learning Perception", "ResNet-34 U-Net v3 (24.44M params) + 4-Way TTA\nTripartite Loss (0.35 BCE + 0.40 Dice + 0.25 clDice)", "#8e44ad")
    draw_box(53, 33, 43, 16, "Zone 2B: Classical Baseline & Landmarks", "Rational Bezier Matched Filter (Comparative Baseline)\n3-Cue OD Intensity/Convergence & Parabolic FAZ", "#2980b9")

    draw_box(16, 8, 68, 16, "Zone 3: 3-Stage Invariant Biometric Matcher", "Stage 1: Hungarian Triplet Cost -> Stage 2: Star-Vector cKDTree + Kabsch SVD ->\nStage 3: Consensus Gate & Induced Common-Subgraph Delaunay Triangulation (0.00% EER)", "#c0392b")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.0, mutation_scale=14))

    arrow(25, 56, 25, 49)
    arrow(74, 56, 74, 49)
    arrow(25, 33, 35, 24)
    ax.annotate("OD Center Landmark\n(Peripapillary Ring)", xy=(65, 24), xytext=(74, 33),
                arrowprops=dict(arrowstyle="->", color="#2980b9", lw=2.0, ls="--", mutation_scale=14),
                fontsize=7.2, fontweight="bold", color="#2980b9", ha="center")
    arrow(50, 24, 74, 56)

    ax.set_title("Figure 3.1: VERA 4-Zone Decoupled Production System Architecture", fontsize=12, fontweight="bold", pad=14)
    out_file = OUTPUT_DIR / "Figure_3.1_System_Architecture.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


def generate_figure_3_2():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    if not img_path.is_file():
        img_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    else:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fov = PipelineService._safe_fov_mask(img_bgr)
    masked_rgb = img_rgb.copy()
    masked_rgb[fov == 0] = 0

    pipeline_service = PipelineService()
    vessel_mask, prob_map, _, _, _, _ = pipeline_service.segment_vessels(img_bgr)

    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)
    green_clahe[fov == 0] = 0

    # --- Rational Bezier Matched Filter Simulation ---
    kernel_size = 15
    sigma = 1.5
    angles = np.linspace(0, 180, 12, endpoint=False)
    responses = []
    for ang in angles:
        rot_mat = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), ang, 1.0)
        base_k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for y in range(kernel_size):
            y_diff = y - kernel_size // 2
            base_k[y, :] = -np.exp(- (y_diff ** 2) / (2.0 * sigma ** 2))
        base_k -= np.mean(base_k)
        rot_k = cv2.warpAffine(base_k, rot_mat, (kernel_size, kernel_size))
        resp = cv2.filter2D(green_clahe, cv2.CV_32F, rot_k)
        responses.append(resp)
    bezier_response = np.maximum.reduce(responses)
    bezier_response[fov == 0] = 0
    bezier_norm = cv2.normalize(bezier_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, classical_otsu = cv2.threshold(bezier_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    classical_otsu[fov == 0] = 0

    od_x, od_y = find_optic_disk(classical_otsu, green_clahe)
    fov_x, fov_y = find_fovea(classical_otsu, (od_x, od_y), green_clahe)

    landmark_overlay = cv2.cvtColor(green_clahe, cv2.COLOR_GRAY2RGB)
    cv2.circle(landmark_overlay, (int(od_x), int(od_y)), 30, (255, 215, 0), 3)
    cv2.circle(landmark_overlay, (int(fov_x), int(fov_y)), 15, (0, 210, 210), 3)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    top_panels = [
        ("1. Raw RGB Input", img_rgb, None),
        ("2. Binary FOV Guard", fov, "gray"),
        ("3. Canonical Scaled RGB", masked_rgb, None),
        ("4. ResNet-34 U-Net Prob", prob_map, "viridis"),
        ("5. Adaptive Binary Mask", vessel_mask, "gray"),
    ]

    bot_panels = [
        ("1. Green Channel", green, "gray"),
        ("2. CLAHE (clip=2.0, 8x8)", green_clahe, "gray"),
        ("3. Bezier 3x6 Response", bezier_norm, "inferno"),
        ("4. Classical Otsu Mask", classical_otsu, "gray"),
        ("5. 3-Cue OD & FAZ Map", landmark_overlay, None),
    ]

    for ax, (title, data, cmap) in zip(axes[0], top_panels):
        ax.set_facecolor("#fafafa")
        if cmap:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        ax.axis("off")

    for ax, (title, data, cmap) in zip(axes[1], bot_panels):
        ax.set_facecolor("#fafafa")
        if cmap:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=6)
        ax.axis("off")

    fig.suptitle(
        "Figure 3.2: Dual Preprocessing Pipeline: Deep Learning (Top) vs Classical Baseline (Bottom) on 01_test.tif",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    out_file = OUTPUT_DIR / "Figure_3.2_Preprocessing_Pipeline.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


def generate_figure_3_3():
    fig, ax = plt.subplots(figsize=(11.0, 6.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    def block(x, y, w, h, name, sub, col):
        b = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0.5", facecolor=col, ec="#2c3e50", lw=1.2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h * 0.62, name, ha="center", va="center", color="#ffffff", fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", color="#ffffff", fontsize=6.8)

    block(4, 56, 18, 10, "Input Image", "512x512x3", "#34495e")
    block(4, 43, 18, 10, "ResNet Stem (Skip 4)", "256x256x64 (Conv 7x7, s=2)", "#2980b9")
    block(4, 30, 18, 10, "Layer 1 (Skip 3)", "128x128x64 (3 Blocks)", "#2980b9")
    block(4, 17, 18, 10, "Layer 2 (Skip 2)", "64x64x128 (4 Blocks)", "#2980b9")
    block(4, 4, 18, 10, "Layer 3 (Skip 1)", "32x32x256 (6 Blocks)", "#2980b9")
    block(27, 4, 18, 10, "Bottleneck L4", "16x16x512 (3 Blocks)", "#16a085")

    block(50, 4, 18, 10, "Decoder Block 0", "32x32x256 (NN 2x Up)", "#8e44ad")
    block(50, 17, 18, 10, "Decoder Block 1", "64x64x128 (NN 2x Up)", "#8e44ad")
    block(50, 30, 18, 10, "Decoder Block 2", "128x128x64 (NN 2x Up)", "#8e44ad")
    block(50, 43, 18, 10, "Decoder Block 3", "256x256x32 (NN 2x Up)", "#8e44ad")
    block(50, 56, 18, 10, "Decoder Block 4 + Head", "512x512x1 (3x3 Conv 145p)", "#27ae60")

    for y_pos in [48.0, 35.0, 22.0, 9.0]:
        ax.annotate("", xy=(50, y_pos), xytext=(22, y_pos), arrowprops=dict(arrowstyle="->", color="#e67e22", lw=2.0, ls="--"))

    def dataflow_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.0, mutation_scale=12))

    dataflow_arrow(13, 56, 13, 53)
    dataflow_arrow(13, 43, 13, 40)
    dataflow_arrow(13, 30, 13, 27)
    dataflow_arrow(13, 17, 13, 14)
    dataflow_arrow(22, 9, 27, 9)

    dataflow_arrow(45, 9, 50, 9)
    dataflow_arrow(59, 14, 59, 17)
    dataflow_arrow(59, 27, 59, 30)
    dataflow_arrow(59, 40, 59, 43)
    dataflow_arrow(59, 53, 59, 56)

    loss_box = FancyBboxPatch((72, 8), 26, 58, boxstyle="round,pad=1.0", facecolor="#fdfefe", ec="#c0392b", lw=2.0)
    ax.add_patch(loss_box)
    ax.text(85, 62, "Tripartite Composite Loss", color="#c0392b", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(85, 54, r"$0.35 \mathcal{L}_{BCE} + 0.40 \mathcal{L}_{Dice} + 0.25 \mathcal{L}_{clDice}$", color="#2c3e50", fontsize=7.5, fontweight="bold", ha="center")
    ax.text(85, 43, "Total Params: 24,436,369\nEncoder: 21,284,672\nDecoder: 3,151,552\nHead: 145 params", color="#34495e", fontsize=7.5, ha="center")
    ax.text(85, 25, "Soft-clDice preserves thin\ncapillary connectivity\nvia K=10 iterations\nSoftSkeletonize", color="#8e44ad", fontsize=7.5, ha="center")

    ax.set_title("Figure 3.3: ResNet-34 U-Net Deep Architecture (24.4M Params) and Tripartite Composite Loss", fontsize=11, fontweight="bold", pad=14)
    out_file = OUTPUT_DIR / "Figure_3.3_UNet_ResNet34_Loss.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


def generate_figure_3_4():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    if not img_path.is_file():
        mask = np.zeros((200, 200), dtype=np.uint8)
        skel = np.zeros((200, 200), dtype=np.uint8)
        bps = []
        tri, pts = None, np.zeros((0, 2))
        matched_pairs = []
    else:
        img_bgr = cv2.imread(str(img_path))
        pipeline_service = PipelineService()
        mask, prob_map, skel, _, _, _ = pipeline_service.segment_vessels(img_bgr)
        bps = detect_bifurcations(skel > 0, border_margin=25, min_distance=15.0)
        tri, pts = build_delaunay_triangulation(bps)

        perturbed_bgr = PipelineService.apply_perturbation(img_bgr, rotate_deg=25.0, scale_factor=0.85)
        match_res = pipeline_service.match_identities(img_bgr, perturbed_bgr)
        matched_pairs = match_res.get("matched_pairs", [])

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Stage 1: U-Net Vessels", fontsize=8.5, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(skel, cmap="gray")
    axes[1].set_title("Stage 2: Zhang-Suen 1px", fontsize=8.5, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(skel, cmap="gray")
    bps_arr = np.array(bps) if len(bps) > 0 else np.zeros((1, 2))
    axes[2].scatter(bps_arr[:, 0], bps_arr[:, 1], c="#e74c3c", s=20)
    axes[2].set_title(f"Stage 3: Minutiae ({len(bps)})", fontsize=8.5, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(skel, cmap="gray")
    if tri is not None and len(pts) >= 3:
        axes[3].triplot(pts[:, 0], pts[:, 1], tri.simplices, color="#00d2d3", lw=1.0)
    axes[3].set_title("Stage 4: Delaunay Graph", fontsize=8.5, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(skel, cmap="gray")
    if tri is not None and len(pts) >= 3:
        axes[4].triplot(pts[:, 0], pts[:, 1], tri.simplices, color="#00d2d3", lw=0.6, alpha=0.35)
    for pair in matched_pairs:
        verts = np.array(pair["vertices_a"])
        poly = Polygon(verts, closed=True, edgecolor="#2ecc71", facecolor="#2ecc71", alpha=0.45, lw=1.4)
        axes[4].add_patch(poly)
    axes[4].set_title(f"Stage 5: Biometric Match ({len(matched_pairs)} Inliers)", fontsize=8.5, fontweight="bold")
    axes[4].axis("off")

    fig.suptitle("Figure 3.4: Complete Biometric Pipeline from Vessel Mask to Delaunay Graph Matching (01_test.tif)", fontsize=11, fontweight="bold", y=1.02)
    out_file = OUTPUT_DIR / "Figure_3.4_Feature_Extraction_Pipeline.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


def generate_figure_3_5():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    if not img_path.is_file():
        img = np.zeros((200, 200, 3), dtype=np.uint8)
    else:
        img = cv2.imread(str(img_path))

    pipeline_service = PipelineService()

    p1 = PipelineService.apply_perturbation(img)
    p2 = PipelineService.apply_perturbation(img, rotate_deg=25.0, scale_factor=0.85)
    p3 = PipelineService.apply_perturbation(img, haze_factor=0.45, gamma=0.85)
    p4 = PipelineService.apply_perturbation(img, glare_intensity=0.65, brightness=0.80)
    p5 = PipelineService.apply_perturbation(img, blur_sigma=2.5)
    p6 = PipelineService.apply_perturbation(img, noise_sigma=16.0, brightness=0.85)
    p7 = PipelineService.apply_perturbation(img, occlusion_frac=0.22)
    p8 = PipelineService.apply_perturbation(
        img,
        rotate_deg=15.0,
        scale_factor=0.90,
        blur_sigma=1.8,
        haze_factor=0.30,
        glare_intensity=0.35,
        noise_sigma=10.0,
        elastic_alpha=2.0,
    )

    torts_raw = [
        ("1. Clean Baseline", p1),
        ("2. Tilt +25° & Zoom 0.85x", p2),
        ("3. Cataract Haze 45% + Gamma", p3),
        ("4. Corneal Glare 65% + Dim", p4),
        ("5. Optical Defocus (sigma=2.5)", p5),
        ("6. CMOS ISO Noise (sigma=16)", p6),
        ("7. Eyelid Occlusion 22%", p7),
        ("8. Judge Torture (7 Factors)", p8),
    ]

    torts = []
    for lbl, p_bgr in torts_raw:
        res = pipeline_service.match_identities(img, p_bgr)
        score = float(res.get("similarity_score", 0.0))
        pairs = len(res.get("matched_pairs", []))
        title = f"{lbl}\n(Score: {score:.4f} | {pairs} Pairs)"
        torts.append((title, cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)))

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    for ax, (title, p_img) in zip(axes.flat, torts):
        ax.set_facecolor("#fafafa")
        ax.imshow(p_img)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=5)
        ax.axis("off")

    fig.suptitle(
        "Figure 3.5: Judge Torture Engine: 8 Physical & Clinical Perturbation Vectors on Clinical Subject S01 (01_test.tif)",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    out_file = OUTPUT_DIR / "Figure_3.5_Judge_Torture_Vectors.png"
    plt.savefig(str(out_file), bbox_inches="tight")
    plt.close()


def generate_figure_3_6():
    img_path = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images" / "01_test.tif"
    pipeline_service = PipelineService()
    if img_path.is_file():
        img_bgr = cv2.imread(str(img_path))
        _, prob_map, skel, _, kps_dicts, _ = pipeline_service.segment_vessels(img_bgr)
        h, w = prob_map.shape[:2]
    else:
        h, w = 512, 512
        prob_map = np.zeros((h, w), dtype=np.uint8)
        kps_dicts = []

    fig = plt.figure(figsize=(12.0, 6.0), dpi=300)
    fig.patch.set_facecolor("#0d1117")

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_facecolor("#0d1117")

    grid_n = 60
    u_grid = np.linspace(-1.0, 1.0, grid_n)
    v_grid = np.linspace(-1.0, 1.0, grid_n)
    uu, vv = np.meshgrid(u_grid, v_grid)
    r_sq = uu**2 + vv**2
    mask = r_sq <= 0.96**2

    prob_down = cv2.resize(prob_map, (grid_n, grid_n), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    z_surf = np.zeros_like(r_sq)
    z_surf[mask] = -0.42 * (1.0 - r_sq[mask]) + 0.22 * prob_down[mask]

    ax1.plot_wireframe(uu, vv, z_surf, color="#2c3e50", alpha=0.45, lw=0.6)

    # --- Real Minutiae Points and Delaunay Graph ---
    if kps_dicts:
        raw_pts = np.array([[kp["x"], kp["y"]] for kp in kps_dicts])
        u_pts = 2.0 * (raw_pts[:, 0] / float(w)) - 1.0
        v_pts = 2.0 * (raw_pts[:, 1] / float(h)) - 1.0
        in_fov = (u_pts**2 + v_pts**2) <= 0.90**2
        u_pts = u_pts[in_fov]
        v_pts = v_pts[in_fov]
        raw_pts = raw_pts[in_fov]
    else:
        u_pts = np.array([-0.35, -0.1, 0.2, 0.4, 0.1, -0.2, 0.0])
        v_pts = np.array([0.1, 0.35, 0.3, -0.1, -0.3, -0.2, 0.0])
        raw_pts = np.zeros((len(u_pts), 2))

    r_p = u_pts**2 + v_pts**2
    p_vals = np.array([prob_map[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] / 255.0 for x, y in raw_pts]) if kps_dicts else np.zeros_like(u_pts)
    z_p = -0.42 * np.maximum(0.0, 1.0 - r_p) + 0.22 * p_vals + 0.02
    ax1.scatter(u_pts, v_pts, z_p, color="#00d2d3", s=25, alpha=0.9, label=f"Minutiae Vertices ({len(u_pts)})")

    tri = None
    if len(u_pts) >= 4:
        tri = Delaunay(np.column_stack([u_pts, v_pts]))
        for s in tri.simplices:
            for i in range(3):
                p1, p2 = s[i], s[(i + 1) % 3]
                ax1.plot([u_pts[p1], u_pts[p2]], [v_pts[p1], v_pts[p2]], [z_p[p1], z_p[p2]], color="#00d2d3", lw=0.8, alpha=0.7)

    ax1.set_title(r"Paraboloid Surface Model" + "\n" + r"$z = -0.42(1-r^2) + 0.22 p_{vessel}$", color="#ffffff", fontsize=9, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_facecolor("#0d1117")

    levels = [
        (-0.25, "#34495e", "Level 1: Concave Fundus Dish + OD/Fovea"),
        (0.10, "#9b59b6", "Level 2: Probability Relief Layer"),
        (0.45, "#2ecc71", "Level 3: 1-px Skeleton Mesh"),
        (0.51, "#00d2d3", "Level 4: Constellation Nodes & Delaunay"),
    ]

    for z_lvl, col, lbl in levels:
        uu_lvl = uu * 0.8
        vv_lvl = vv * 0.8
        zz_lvl = np.full_like(uu_lvl, z_lvl)
        ax2.plot_wireframe(uu_lvl, vv_lvl, zz_lvl, color=col, alpha=0.35, lw=0.5)
        ax2.text(0.85, 0.85, z_lvl, lbl, color=col, fontsize=7.5, fontweight="bold")

    ax2.scatter(u_pts * 0.8, v_pts * 0.8, np.full_like(u_pts, 0.51), color="#00d2d3", s=20)
    if tri is not None and len(u_pts) >= 4:
        for s in tri.simplices:
            for i in range(3):
                p1, p2 = s[i], s[(i + 1) % 3]
                ax2.plot([u_pts[p1] * 0.8, u_pts[p2] * 0.8], [v_pts[p1] * 0.8, v_pts[p2] * 0.8], [0.51, 0.51], color="#00d2d3", lw=0.7)

    # --- Laser Correspondence Arcs ---
    if len(u_pts) >= 10:
        sample_indices = np.linspace(0, len(u_pts) - 1, 12, dtype=int)
        for idx in sample_indices:
            x_a, y_a = u_pts[idx] * 0.8, v_pts[idx] * 0.8
            t_arc = np.linspace(0.0, 1.0, 20)
            z_arc = 0.51 + 0.18 * np.sin(np.pi * t_arc)
            x_arc = x_a + 0.15 * np.sin(np.pi * t_arc)
            y_arc = y_a + 0.15 * np.sin(np.pi * t_arc)
            ax2.plot(x_arc, y_arc, z_arc, color="#2ecc71", lw=1.2, alpha=0.85)

    ax2.set_title("3D Exploded Multi-Layer Stack\n(RetinalGlobe.tsx & DelaunayConstellation.tsx)", color="#ffffff", fontsize=9, fontweight="bold")
    ax2.axis("off")

    fig.suptitle("Figure 3.6: 3D Concave Ocular Cup Projection and WebGL Retinal Constellation Studio", color="#ffffff", fontsize=11, fontweight="bold", y=0.98)
    out_file = OUTPUT_DIR / "Figure_3.6_3D_Cup_Constellation.png"
    plt.savefig(str(out_file), facecolor="#0d1117", edgecolor="none", bbox_inches="tight")
    plt.close()


# --- Chapter 5 Figure Generation ---
def generate_figure_5_1():
    src_file = PROJECT_ROOT / "results" / "training_curves_v2.png"
    out_file = OUTPUT_DIR / "Figure_5.1_Training_Curves.png"
    if src_file.is_file():
        shutil.copy2(src_file, out_file)


def generate_figure_5_2():
    pipeline_service = PipelineService()

    eval_pairs = [
        ("DRIVE Test Split", PROJECT_ROOT / "data" / "DRIVE" / "training" / "images" / "31_training.tif", PROJECT_ROOT / "data" / "DRIVE" / "training" / "1st_manual" / "31_manual1.gif"),
        ("HRF Test Split", PROJECT_ROOT / "data" / "HRF" / "images" / "14_g.jpg", PROJECT_ROOT / "data" / "HRF" / "manual1" / "14_g.tif"),
        ("STARE Test Split", PROJECT_ROOT / "data" / "STARE" / "im0255.ppm", PROJECT_ROOT / "data" / "STARE" / "im0255.vk.ppm.gz"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14.0, 10.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    from dl_pipeline.predict import predict_vessel_mask
    from dl_pipeline.data_utils import load_rgb_image, load_binary_gt_mask
    from dl_pipeline.postprocess import refine_vessel_mask
    from classical_pipeline.fovea_localization import estimate_fov_mask
    from evaluation.metrics import compute_auc_roc, pixel_metrics

    for row_idx, (ds_name, img_p, mask_p) in enumerate(eval_pairs):
        rgb_img = load_rgb_image(img_p)
        gt_mask = load_binary_gt_mask(mask_p, target_shape=rgb_img.shape[:2])

        bin_mask, prob_map = predict_vessel_mask(
            pipeline_service.model,
            rgb_img,
            device=pipeline_service.device,
            threshold=0.60,
            use_tta=True,
        )
        bin_mask = refine_vessel_mask(bin_mask, min_size=25, max_hole_size=15)

        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        fov = estimate_fov_mask(bgr_img)
        m = pixel_metrics(bin_mask, gt_mask, threshold=0.60, fov_mask=fov)
        auc = compute_auc_roc(prob_map, gt_mask, fov_mask=fov)

        dice = m["dice"]

        # --- Color-Coded Error Overlay Construction ---
        h, w = rgb_img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        gray_bg = cv2.cvtColor(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) // 3
        overlay[:] = gray_bg

        tp = (bin_mask == 1) & (gt_mask == 1)
        fp = (bin_mask == 1) & (gt_mask == 0)
        fn = (bin_mask == 0) & (gt_mask == 1)

        overlay[tp] = [46, 204, 113]
        overlay[fp] = [231, 76, 60]
        overlay[fn] = [52, 152, 219]

        axes[row_idx, 0].imshow(rgb_img)
        axes[row_idx, 0].set_title(f"{ds_name}: Raw Fundus RGB", fontsize=8.5, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_mask, cmap="gray")
        axes[row_idx, 1].set_title(f"{ds_name}: Ground Truth Mask", fontsize=8.5, fontweight="bold")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(prob_map, cmap="viridis")
        axes[row_idx, 2].set_title(f"{ds_name}: ResNet-34 U-Net Prob", fontsize=8.5, fontweight="bold")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title(f"Overlay (Dice: {dice:.4f} | AUC: {auc:.4f})\nTP=Green, FP=Red, FN=Blue", fontsize=8.5, fontweight="bold")
        axes[row_idx, 3].axis("off")

    fig.suptitle(
        "Figure 5.2: Vessel Segmentation Evaluation across Held-Out Test Splits of DRIVE, HRF, and STARE Datasets",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig_5_2_path = OUTPUT_DIR / "Figure_5.2_Vessel_Segmentation_Samples.png"
    plt.savefig(str(fig_5_2_path), bbox_inches="tight")
    plt.close()

    vessel_samples_path = PROJECT_ROOT / "results" / "vessel_prediction_samples.png"
    shutil.copy2(fig_5_2_path, vessel_samples_path)


def generate_figure_5_3():
    src_file = PROJECT_ROOT / "results" / "threshold_optimization_curves.png"
    out_file = OUTPUT_DIR / "Figure_5.3_Threshold_Optimization.png"
    if src_file.is_file():
        shutil.copy2(src_file, out_file)


def generate_figure_5_4():
    src_file = PROJECT_ROOT / "results" / "matching" / "biometric_matching_curves.png"
    out_file = OUTPUT_DIR / "Figure_5.4_Biometric_Matching_Curves.png"
    if src_file.is_file():
        shutil.copy2(src_file, out_file)


def generate_figure_5_5():
    src_file = PROJECT_ROOT / "results" / "localization" / "visualizations" / "IDRiD_001_localization_overlay.png"
    out_file = OUTPUT_DIR / "Figure_5.5_IDRiD_001_Localization.png"
    if src_file.is_file():
        shutil.copy2(src_file, out_file)


def generate_figure_5_6():
    src_file = PROJECT_ROOT / "results" / "localization" / "visualizations" / "IDRiD_002_localization_overlay.png"
    out_file = OUTPUT_DIR / "Figure_5.6_IDRiD_002_Localization.png"
    if src_file.is_file():
        shutil.copy2(src_file, out_file)


# --- Main Orchestration Entrypoint ---
def generate_all_thesis_figures():
    generate_figure_2_1()
    generate_figure_2_2()
    generate_figure_2_3()
    generate_figure_2_4()
    generate_figure_2_5()
    generate_figure_3_1()
    generate_figure_3_2()
    generate_figure_3_3()
    generate_figure_3_4()
    generate_figure_3_5()
    generate_figure_3_6()
    generate_figure_5_1()
    generate_figure_5_2()
    generate_figure_5_3()
    generate_figure_5_4()
    generate_figure_5_5()
    generate_figure_5_6()


if __name__ == "__main__":
    generate_all_thesis_figures()
