import ast
from pathlib import Path
import re
import tokenize

import cv2
import numpy as np


# --- Character & Emoji Validation Helpers ---
def _is_emoji_char(char: str) -> bool:
    code_point = ord(char)
    return (
        (0x1F600 <= code_point <= 0x1F64F)
        or (0x1F300 <= code_point <= 0x1F5FF)
        or (0x1F680 <= code_point <= 0x1F6FF)
        or (0x1F700 <= code_point <= 0x1F77F)
        or (0x1F780 <= code_point <= 0x1F7FF)
        or (0x1F800 <= code_point <= 0x1F8FF)
        or (0x1F900 <= code_point <= 0x1F9FF)
        or (0x1FA00 <= code_point <= 0x1FA6F)
        or (0x1FA70 <= code_point <= 0x1FAFF)
        or (0x2600 <= code_point <= 0x26FF)
        or (0x2700 <= code_point <= 0x27BF)
    )


# --- Codebase Syntax & Compilation Verification ---
def test_codebase_syntax_and_compilation():
    root = Path(__file__).resolve().parent.parent
    subdirs = ["api", "classical_pipeline", "dl_pipeline", "evaluation", "matching", "tests"]
    py_files: list[Path] = []
    for d in subdirs:
        dir_path = root / d
        if dir_path.is_dir():
            py_files.extend(dir_path.rglob("*.py"))

    assert len(py_files) > 20, f"Expected > 20 Python files, discovered {len(py_files)}"

    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(py_file))
            assert tree is not None
        except Exception as e:
            raise AssertionError(f"AST parse failed for {py_file}: {e}") from e

        try:
            code_obj = compile(content, str(py_file), "exec")
            assert code_obj is not None
        except Exception as e:
            raise AssertionError(f"compile() failed for {py_file}: {e}") from e


# --- Universal Zero-Emoji Validation Across Python Files ---
def test_zero_emojis_in_python_codebase():
    root = Path(__file__).resolve().parent.parent
    subdirs = ["api", "classical_pipeline", "dl_pipeline", "evaluation", "matching", "tests"]
    for d in subdirs:
        dir_path = root / d
        if dir_path.is_dir():
            for py_file in dir_path.rglob("*.py"):
                content = py_file.read_text(encoding="utf-8")
                for line_idx, line in enumerate(content.splitlines(), 1):
                    for char in line:
                        assert not _is_emoji_char(char), (
                            f"Emoji character '{char}' (U+{ord(char):04X}) detected in {py_file}:{line_idx}"
                        )


# --- Static Analysis: Zero Unused Imports ---
def test_no_unused_imports_in_python_codebase():
    root = Path(__file__).resolve().parent.parent
    subdirs = ["api", "classical_pipeline", "dl_pipeline", "evaluation", "matching", "tests"]

    for d in subdirs:
        dir_path = root / d
        if not dir_path.is_dir():
            continue
        for py_file in dir_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            imported: dict[str, int] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported[a.asname or a.name.split(".")[0]] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "__future__":
                        continue
                    for a in node.names:
                        if a.name != "*":
                            imported[a.asname or a.name] = node.lineno

            used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    curr = node.value
                    while isinstance(curr, ast.Attribute):
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        used.add(curr.id)

            unused = [f"{n} (line {l})" for n, l in imported.items() if n not in used]
            assert not unused, f"Unused imports detected in {py_file}: {unused}"


# --- Zero-Emoji Policy Across Documentation & Web Frontend ---
def test_zero_emojis_across_docs_and_web():
    root = Path(__file__).resolve().parent.parent
    files_to_check: list[Path] = []

    for doc in ["README.md", "ROADMAP.md"]:
        p = root / doc
        if p.is_file():
            files_to_check.append(p)

    results_dir = root / "results"
    if results_dir.is_dir():
        files_to_check.extend(results_dir.rglob("*.md"))

    web_src = root / "web" / "src"
    if web_src.is_dir():
        for ext in [".ts", ".tsx", ".css"]:
            files_to_check.extend(web_src.rglob(f"*{ext}"))

    docs_dir = root / "docs"
    if docs_dir.is_dir():
        for ext in [".md", ".bib"]:
            files_to_check.extend(docs_dir.rglob(f"*{ext}"))

    for f in files_to_check:
        content = f.read_text(encoding="utf-8")
        for line_idx, line in enumerate(content.splitlines(), 1):
            for char in line:
                assert not _is_emoji_char(char), (
                    f"Emoji character '{char}' (U+{ord(char):04X}) detected in {f}:{line_idx}"
                )


# --- Strict Comment Format Policy Validation ---
def test_strict_comment_policy_across_codebase():
    root = Path(__file__).resolve().parent.parent
    subdirs = ["api", "classical_pipeline", "dl_pipeline", "evaluation", "matching", "tests"]
    py_pattern = re.compile(r"^#\s*---\s*.+\s*---\s*$|^#\s*(type:|noqa|pragma:)|^#!\s*/")

    py_violations: list[str] = []
    for d in subdirs:
        dir_path = root / d
        if not dir_path.is_dir():
            continue
        for py_file in dir_path.rglob("*.py"):
            with open(py_file, "rb") as f:
                tokens = tokenize.tokenize(f.readline)
                for tok in tokens:
                    if tok.type == tokenize.COMMENT:
                        val = tok.string.strip()
                        if not py_pattern.match(val):
                            py_violations.append(f"{py_file.name}:{tok.start[0]}: {val}")

    assert not py_violations, f"Non-section Python comments detected: {py_violations}"

    ts_pattern = re.compile(r"^\s*//\s*---\s*.+\s*---\s*$|^\s*//\s*(eslint-disable|@ts-)")
    ts_violations: list[str] = []
    web_src = root / "web" / "src"
    if web_src.is_dir():
        for f in web_src.rglob("*"):
            if f.suffix in (".ts", ".tsx"):
                for idx, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    stripped = line.strip()
                    if stripped.startswith("//"):
                        if not ts_pattern.match(stripped):
                            ts_violations.append(f"{f.name}:{idx}: {stripped}")

    assert not ts_violations, f"Non-section TypeScript comments detected: {ts_violations}"


# --- Thesis Manuscript and Scientific Figures Integrity ---
def test_thesis_manuscript_integrity():
    root = Path(__file__).resolve().parent.parent
    manuscript = root / "docs" / "VERA_THESIS_MANUSCRIPT.md"
    assert manuscript.is_file(), "Missing docs/VERA_THESIS_MANUSCRIPT.md"

    text = manuscript.read_text(encoding="utf-8")

    for line_idx, line in enumerate(text.splitlines(), 1):
        for char in line:
            assert not _is_emoji_char(char), (
                f"Emoji character '{char}' (U+{ord(char):04X}) detected in manuscript:{line_idx}"
            )

    chapters = re.findall(r"^#+\s+บทที่\s+(\d+)", text, re.MULTILINE)
    assert len(chapters) == len(set(chapters)), f"Duplicate chapter numbers found in manuscript: {chapters}"

    figures_dir = root / "results" / "figures"
    assert figures_dir.is_dir(), "Figures directory results/figures does not exist"

    expected_figures = [
        "Figure_2.1_Retinal_Anatomy.png",
        "Figure_2.2_Hemoglobin_Spectrum.png",
        "Figure_2.3_Rutovitz_Crossing_Number.png",
        "Figure_2.4_Kabsch_SVD_Alignment.png",
        "Figure_2.5_Delaunay_Triangulation.png",
        "Figure_3.1_System_Architecture.png",
        "Figure_3.2_Preprocessing_Pipeline.png",
        "Figure_3.3_UNet_ResNet34_Loss.png",
        "Figure_3.4_Feature_Extraction_Pipeline.png",
        "Figure_3.5_Judge_Torture_Vectors.png",
        "Figure_3.6_3D_Cup_Constellation.png",
        "Figure_5.1_Training_Curves.png",
        "Figure_5.2_Vessel_Segmentation_Samples.png",
        "Figure_5.3_Threshold_Optimization.png",
        "Figure_5.4_Biometric_Matching_Curves.png",
        "Figure_5.5_IDRiD_001_Localization.png",
        "Figure_5.6_IDRiD_002_Localization.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.is_file(), f"Missing required thesis figure: {fig_name}"
        assert fig_path.stat().st_size > 15_000, f"Figure {fig_name} is suspiciously small ({fig_path.stat().st_size} bytes)"
        img = cv2.imread(str(fig_path))
        assert img is not None and img.shape[0] >= 600 and img.shape[1] >= 800, f"Invalid resolution for {fig_name}"

    # --- Figure Content Verification ---
    fig_3_2 = figures_dir / "Figure_3.2_Preprocessing_Pipeline.png"
    img_3_2 = cv2.imread(str(fig_3_2))
    assert img_3_2 is not None
    std_channels = [float(np.std(img_3_2[:, :, c])) for c in range(3)]
    assert all(s > 18.0 for s in std_channels), f"Figure 3.2 lacks expected channel variance: {std_channels}"

    fig_3_4 = figures_dir / "Figure_3.4_Feature_Extraction_Pipeline.png"
    img_3_4 = cv2.imread(str(fig_3_4))
    assert img_3_4 is not None
    hsv_3_4 = cv2.cvtColor(img_3_4, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_3_4, np.array([40, 50, 50]), np.array([85, 255, 255]))
    green_pixels = int(np.count_nonzero(green_mask))
    assert green_pixels > 300, f"Figure 3.4 lacks emerald green inlier triangles: {green_pixels} px"

    for match in re.finditer(r"!\[.*?\]\(((\.\./)?results/figures/([^)]+))\)", text):
        fig_target = figures_dir / match.group(3)
        assert fig_target.is_file(), f"Manuscript references non-existent figure: {match.group(1)}"

    bib_split = re.split(r"#+\s+บรรณานุกรม", text)
    assert len(bib_split) >= 2, "Bibliography section not found in manuscript"
    body_text = bib_split[0]
    bib_text = bib_split[1]

    body_citations: set[int] = set()
    for citation_match in re.finditer(r"\[([0-9,\s\-–]+)\]", body_text):
        token = citation_match.group(1).strip()
        parts = token.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part or "–" in part:
                range_tokens = re.split(r"[-–]", part)
                if len(range_tokens) == 2 and range_tokens[0].strip().isdigit() and range_tokens[1].strip().isdigit():
                    s_idx = int(range_tokens[0].strip())
                    e_idx = int(range_tokens[1].strip())
                    for c_num in range(s_idx, e_idx + 1):
                        body_citations.add(c_num)
            elif part.isdigit():
                c_val = int(part)
                if c_val > 0:
                    body_citations.add(c_val)

    bib_entries = {int(m) for m in re.findall(r"^\[(\d+)\]", bib_text, re.MULTILINE)}
    assert body_citations == bib_entries, (
        f"Citation bijection mismatch: in body only: {body_citations - bib_entries}, in bib only: {bib_entries - body_citations}"
    )
    assert bib_entries == set(range(1, len(bib_entries) + 1)), (
        f"Bibliography entries must be contiguous 1..N without gaps: {sorted(bib_entries)}"
    )
