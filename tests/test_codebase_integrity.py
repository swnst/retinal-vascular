import ast
from pathlib import Path
import re
import tokenize


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
        # --- Syntax Verification via AST ---
        try:
            tree = ast.parse(content, filename=str(py_file))
            assert tree is not None
        except Exception as e:
            raise AssertionError(f"AST parse failed for {py_file}: {e}") from e

        # --- Bytecode Compilation Verification ---
        try:
            code_obj = compile(content, str(py_file), "exec")
            assert code_obj is not None
        except Exception as e:
            raise AssertionError(f"compile() failed for {py_file}: {e}") from e


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
