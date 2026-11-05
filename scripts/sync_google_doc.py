import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# --- Paths & Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_ID = "1VboeJdU93dWi41VzIkhnn5OP_jjC7TsVElagkpIK75Y"
MANUSCRIPT_PATH = PROJECT_ROOT / "docs" / "VERA_THESIS_MANUSCRIPT.md"
FIGURES_JSON = PROJECT_ROOT / "results" / "drive_uploaded_figures.json"
OUTPUT_HTML = PROJECT_ROOT / "results" / "Proเจ๊ง_manuscript.html"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
CHECK_HTML = SCRATCH_DIR / "exported_doc_check.html"

GDRIVE_BIN = shutil.which("gdrive") or "gdrive"


# --- Math Typography Transformation ---
def clean_latex_math(text: str) -> str:
    replacements = [
        (r"\\mathbb\{R\}\^2", "ℝ²"),
        (r"\\mathbb\{R\}", "ℝ"),
        (r"\\mathbb\{Z\}\^2", "ℤ²"),
        (r"\\mathbb\{Z\}", "ℤ"),
        (r"SO\(2\)", "SO(2)"),
        (r"\\text\{SO\}\(2\)", "SO(2)"),
        (r"\\text\{diag\}", "diag"),
        (r"\\text\{det\}", "det"),
        (r"\\text\{Tr\}", "Tr"),
        (r"\\text\{MAD\}", "MAD"),
        (r"\\text\{vessel\}", "vessel"),
        (r"\\text\{adj\}", "adj"),
        (r"\\text\{min\}", "min"),
        (r"\\text\{max\}", "max"),
        (r"\\text\{deg\}", "deg"),
        (r"\\in", "∈"),
        (r"\\notin", "∉"),
        (r"\\le", "≤"),
        (r"\\ge", "≥"),
        (r"\\neq", "≠"),
        (r"\\approx", "≈"),
        (r"\\times", "×"),
        (r"\\cdot", "·"),
        (r"\\pm", "±"),
        (r"\\sum", "∑"),
        (r"\\prod", "∏"),
        (r"\\partial", "∂"),
        (r"\\nabla", "∇"),
        (r"\\sigma", "σ"),
        (r"\\mu", "μ"),
        (r"\\lambda", "λ"),
        (r"\\theta", "θ"),
        (r"\\phi", "φ"),
        (r"\\Phi", "Φ"),
        (r"\\alpha", "α"),
        (r"\\beta", "β"),
        (r"\\gamma", "γ"),
        (r"\\epsilon", "ε"),
        (r"\\delta", "δ"),
        (r"\\Delta", "Δ"),
        (r"\\wedge", "∧"),
        (r"\\lor", "∨"),
        (r"\\forall", "∀"),
        (r"\\exists", "∃"),
        (r"\\to", "→"),
        (r"\\rightarrow", "→"),
        (r"\\infty", "∞"),
        (r"\\sqrt\{([^}]+)\}", r"√(\1)"),
        (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1 / \2)"),
        (r"\\text\{([^}]+)\}", r"\1"),
        (r"\\mathbf\{([^}]+)\}", r"<b>\1</b>"),
        (r"\\hat\{([^}]+)\}", r"\1̂"),
        (r"\\bar\{([^}]+)\}", r"\1̄"),
        (r"\\vec\{([^}]+)\}", r"\1⃗"),
        (r"\\left\(", "("),
        (r"\\right\)", ")"),
        (r"\\left\[", "["),
        (r"\\right\]", "]"),
        (r"\\left\|", "‖"),
        (r"\\right\|", "‖"),
        (r"\\left\\{", "{"),
        (r"\\right\\}", "}"),
        (r"\_\{([^}]+)\}", r"_\1"),
        (r"\^\{([^}]+)\}", r"^\1"),
    ]
    res = text
    for pat, rep in replacements:
        res = re.sub(pat, rep, res)
    return res


# --- Markdown to Clean Academic HTML Converter ---
def convert_markdown_to_html(md_text: str, figures_map: dict[str, dict[str, str]]) -> str:
    lines = md_text.splitlines()
    html_out: list[str] = []
    in_code_block = False
    code_block_lines: list[str] = []
    in_table = False
    table_lines: list[str] = []

    def flush_table():
        nonlocal in_table, table_lines
        if not table_lines:
            in_table = False
            return
        t_html = ['<table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-family: Sarabun, sans-serif; font-size: 10pt;">']
        is_header = True
        for row in table_lines:
            clean_row = row.strip()
            if not clean_row.startswith("|"):
                continue
            cells = [c.strip() for c in clean_row.split("|")[1:-1]]
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                is_header = False
                continue
            t_html.append("  <tr>")
            for c in cells:
                c_content = parse_inline(c)
                if is_header:
                    t_html.append(f'    <th style="border-top: 2px solid #222; border-bottom: 2px solid #222; padding: 10px 14px; text-align: left; background-color: #f1f3f5; font-weight: bold;">{c_content}</th>')
                else:
                    t_html.append(f'    <td style="border-bottom: 1px solid #e2e8f0; padding: 9px 14px; text-align: left;">{c_content}</td>')
            t_html.append("  </tr>")
        t_html.append("</table>")
        html_out.append("\n".join(t_html))
        table_lines = []
        in_table = False

    def parse_inline(text: str) -> str:
        s = text
        # Images
        img_match = re.search(r"!\[(.*?)\]\((.*?)\)", s)
        if img_match:
            alt, src = img_match.group(1), img_match.group(2)
            fname = Path(src).name
            fig_data = figures_map.get(fname)
            if fig_data:
                lh3_url = fig_data["lh3_url"]
                return f'<div style="text-align: center; margin: 26px 0;"><img src="{lh3_url}" alt="{alt}" width="650" style="max-width: 100%; height: auto; border: 1px solid #cbd5e1; border-radius: 4px;" /><br /><p style="font-size: 9.5pt; color: #334155; font-weight: bold; margin-top: 8px;">{alt}</p></div>'
            else:
                return f'<p style="color: red; font-weight: bold;">[Missing Figure: {fname}]</p>'

        # Inline math $...$
        s = re.sub(r"\$([^\$]+)\$", lambda m: f'<span style="font-family: \'Cambria Math\', \'Times New Roman\', serif; font-style: italic;">{clean_latex_math(m.group(1))}</span>', s)
        # Bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        # Italic
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        # Inline code
        s = re.sub(r"`([^`]+)`", r'<code style="background-color: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: Consolas, monospace; font-size: 9.5pt; color: #0f172a;">\1</code>', s)
        return s

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Code block toggle
        if stripped.startswith("```"):
            if in_code_block:
                code_content = "\n".join(code_block_lines)
                html_out.append(f'<pre style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; font-family: Consolas, monospace; font-size: 9.5pt; overflow-x: auto; line-height: 1.5; color: #1e293b;"><code>{code_content}</code></pre>')
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            i += 1
            continue

        # Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        if not stripped:
            i += 1
            continue

        # Display math $$ ... $$
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) >= 4:
                raw_math = stripped[2:-2].strip()
                cleaned = clean_latex_math(raw_math)
                html_out.append(f'<div style="text-align: center; margin: 18px 0; padding: 12px; background-color: #f8fafc; border-left: 3px solid #3b82f6; font-family: \'Cambria Math\', \'Times New Roman\', serif; font-size: 11pt; font-style: italic; line-height: 1.6; color: #0f172a;">{cleaned}</div>')
                i += 1
                continue
            else:
                math_lines = [stripped[2:].strip()] if len(stripped) > 2 else []
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i])
                    i += 1
                if i < n and lines[i].strip().endswith("$$"):
                    tail = lines[i].strip()[:-2].strip()
                    if tail:
                        math_lines.append(tail)
                    i += 1
                raw_math = "\n".join(math_lines).strip()
                cleaned = clean_latex_math(raw_math)
                html_out.append(f'<div style="text-align: center; margin: 18px 0; padding: 12px; background-color: #f8fafc; border-left: 3px solid #3b82f6; font-family: \'Cambria Math\', \'Times New Roman\', serif; font-size: 11pt; font-style: italic; line-height: 1.6; color: #0f172a;">{cleaned}</div>')
                continue

        # Image alone
        if "![" in stripped and "](" in stripped:
            html_out.append(parse_inline(stripped))
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            html_out.append(f'<h1 style="font-family: Sarabun, sans-serif; font-size: 20pt; font-weight: 800; color: #0f172a; margin-top: 36px; margin-bottom: 16px; border-bottom: 2px solid #0f172a; padding-bottom: 8px;">{parse_inline(stripped[2:])}</h1>')
        elif stripped.startswith("## "):
            html_out.append(f'<h2 style="font-family: Sarabun, sans-serif; font-size: 16pt; font-weight: 700; color: #1e293b; margin-top: 28px; margin-bottom: 12px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px;">{parse_inline(stripped[3:])}</h2>')
        elif stripped.startswith("### "):
            html_out.append(f'<h3 style="font-family: Sarabun, sans-serif; font-size: 13pt; font-weight: 700; color: #334155; margin-top: 22px; margin-bottom: 10px;">{parse_inline(stripped[4:])}</h3>')
        elif stripped.startswith("#### "):
            html_out.append(f'<h4 style="font-family: Sarabun, sans-serif; font-size: 11.5pt; font-weight: 700; color: #475569; margin-top: 18px; margin-bottom: 8px;">{parse_inline(stripped[5:])}</h4>')
        # Horizontal rule
        elif stripped in ("---", "***", "___"):
            html_out.append('<hr style="border: none; border-top: 1px solid #cbd5e1; margin: 24px 0;" />')
        # Blockquotes
        elif stripped.startswith(">"):
            html_out.append(f'<blockquote style="margin: 14px 0; padding: 10px 18px; border-left: 4px solid #64748b; background-color: #f8fafc; color: #334155; font-style: italic;">{parse_inline(stripped[1:].strip())}</blockquote>')
        # Bullet list
        elif stripped.startswith("- ") or stripped.startswith("* "):
            html_out.append(f'<p style="margin: 4px 0 4px 24px; text-indent: -16px; font-family: Sarabun, sans-serif; font-size: 11pt; line-height: 1.6; color: #1e293b;">• {parse_inline(stripped[2:])}</p>')
        # Numbered list
        elif re.match(r"^\d+\.\s", stripped):
            num_match = re.match(r"^(\d+\.)\s*(.*)$", stripped)
            prefix = num_match.group(1)
            content = num_match.group(2)
            html_out.append(f'<p style="margin: 4px 0 4px 24px; text-indent: -16px; font-family: Sarabun, sans-serif; font-size: 11pt; line-height: 1.6; color: #1e293b;">{prefix} {parse_inline(content)}</p>')
        # Normal paragraph
        else:
            html_out.append(f'<p style="margin: 10px 0; text-align: justify; text-justify: inter-word; font-family: Sarabun, sans-serif; font-size: 11pt; line-height: 1.7; color: #1e293b;">{parse_inline(stripped)}</p>')

        i += 1

    if in_table:
        flush_table()

    body_content = "\n".join(html_out)
    full_html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8" />
<title>VERA Master Thesis Manuscript</title>
<style>
  body {{
    font-family: 'Sarabun', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #1e293b;
    line-height: 1.7;
    font-size: 11pt;
    margin: 40px auto;
    max-width: 900px;
    padding: 0 20px;
  }}
  h1, h2, h3, h4 {{
    font-family: 'Sarabun', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #0f172a;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 24px 0;
  }}
  th, td {{
    padding: 10px 14px;
    text-align: left;
  }}
  img {{
    max-width: 100%;
    height: auto;
  }}
</style>
</head>
<body>
{body_content}
</body>
</html>"""
    return full_html


# --- Main Synchronization Workflow ---
def run_sync_google_doc():
    if not FIGURES_JSON.is_file():
        print(f"ERROR: {FIGURES_JSON} does not exist. Run sync_cloud_figures.py first.", file=sys.stderr)
        sys.exit(1)

    figures_map = json.loads(FIGURES_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(figures_map)} uploaded figures from mapping.")

    md_text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    print(f"Read manuscript from {MANUSCRIPT_PATH} ({len(md_text.splitlines())} lines).")

    html_content = convert_markdown_to_html(md_text, figures_map)
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    print(f"Generated clean academic HTML at {OUTPUT_HTML} ({len(html_content)} chars).")

    # Update Google Doc in-place
    print(f"Updating Google Doc ID {DOC_ID}...")
    up_cmd = [
        GDRIVE_BIN, "mutate", "update", DOC_ID, str(OUTPUT_HTML),
        "--mime-type", "application/vnd.google-apps.document"
    ]
    up_res = subprocess.run(up_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, shell=True)
    print("Google Doc updated successfully.")
    print(up_res.stdout.strip())

    # Export back and verify images count via HTML and DOCX
    print("Exporting Google Doc back from Drive as HTML to verify image count...")
    check_html = SCRATCH_DIR / "exported_doc_check.html"
    export_html_cmd = [
        GDRIVE_BIN, "readonly", "export", DOC_ID, str(check_html),
        "--format", "html"
    ]
    res_html = subprocess.run(export_html_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", shell=True)

    img_count = 0
    if check_html.is_file() and res_html.returncode == 0:
        html_txt = check_html.read_text(encoding="utf-8", errors="replace")
        img_count = len(re.findall(r"<img\b", html_txt, re.IGNORECASE))
        print(f"HTML Export Verification: Exported HTML contains {img_count} images.")
    else:
        print(f"HTML Export notice: {res_html.stderr.strip() or 'failed'}, checking DOCX...")

    check_docx = SCRATCH_DIR / "exported_doc_check.docx"
    export_docx_cmd = [
        GDRIVE_BIN, "readonly", "export", DOC_ID, str(check_docx),
        "--format", "docx"
    ]
    subprocess.run(export_docx_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, shell=True)

    import zipfile
    with zipfile.ZipFile(check_docx, "r") as z:
        media_files = [f for f in z.namelist() if f.startswith("word/media/")]
        docx_img_count = len(media_files)

    print(f"DOCX Archive Verification: Exported document contains {docx_img_count} images.")

    if img_count == 17 or docx_img_count == 17:
        print("PERFECT: Exactly 17 figures successfully rendered and embedded in Google Doc.")
    else:
        print(f"WARNING: Expected 17 figures, found HTML: {img_count}, DOCX: {docx_img_count}")


if __name__ == "__main__":
    run_sync_google_doc()
