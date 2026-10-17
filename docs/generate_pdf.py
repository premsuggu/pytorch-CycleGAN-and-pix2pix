import os
import sys
import re
import subprocess
from pathlib import Path
import markdown

def convert_markdown_to_html(md_text, base_dir):
    # 1. Protect Math Blocks from Markdown Escaping
    math_placeholders = []
    def stash_math(match):
        idx = len(math_placeholders)
        math_placeholders.append(match.group(0))
        return f"XMATHTOKEN{idx}X"

    # Match display math $$...$$
    md_processed = re.sub(r'\$\$.*?\$\$', stash_math, md_text, flags=re.DOTALL)
    # Match inline math $...$
    md_processed = re.sub(r'(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)', stash_math, md_processed)

    # 2. Process Mermaid blocks
    def replace_mermaid(match):
        code = match.group(1).strip()
        return f'<div class="mermaid-container"><div class="mermaid">\n{code}\n</div></div>'
    
    md_processed = re.sub(r'```mermaid\s*\n(.*?)\n```', replace_mermaid, md_processed, flags=re.DOTALL)
    
    # 3. Process Callouts / Alert Blocks
    def replace_alerts(match):
        alert_type = match.group(1).upper()
        content = match.group(2).strip()
        # Clean leading > marks
        content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
        icon_map = {
            'IMPORTANT': '⚠️ <strong>IMPORTANT:</strong>',
            'WARNING': '🚨 <strong>WARNING:</strong>',
            'NOTE': 'ℹ️ <strong>NOTE:</strong>',
            'TIP': '💡 <strong>TIP:</strong>'
        }
        prefix = icon_map.get(alert_type, f'<strong>{alert_type}:</strong>')
        return f'<div class="alert alert-{alert_type.lower()}">{prefix} {content}</div>'

    md_processed = re.sub(r'>\s*\[!(IMPORTANT|WARNING|NOTE|TIP)\]\s*\n((?:>.*\n?)+)', replace_alerts, md_processed)

    # 4. Markdown Conversion
    extensions = ['tables', 'fenced_code', 'toc', 'sane_lists']
    html_body = markdown.markdown(md_processed, extensions=extensions)
    
    # 5. Restore Math Blocks
    for idx, math_content in enumerate(math_placeholders):
        html_body = html_body.replace(f"XMATHTOKEN{idx}X", math_content)

    # 6. Resolve relative image paths
    def fix_img_src(match):
        src = match.group(1)
        if not src.startswith(('http://', 'https://', 'file://', '/')):
            abs_p = (base_dir / src).resolve()
            return f'src="file://{abs_p}"'
        elif src.startswith('/'):
            return f'src="file://{src}"'
        return match.group(0)

    html_body = re.sub(r'src=["\'](.*?)["\']', fix_img_src, html_body)

    # 7. Wrap Tables for Page-Break Control
    html_body = re.sub(r'(<table>.*?</table>)', r'<div class="table-container">\1</div>', html_body, flags=re.DOTALL)

    # 8. Full HTML Document with Publication-Grade Styles
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Topology-Preserving Cross-Modality Retinal Image Translation</title>
<style>
@page {{
    size: A4;
    margin: 22mm 18mm 22mm 18mm;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a1a1a;
    line-height: 1.6;
    font-size: 9.5pt;
    background: #fff;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: 19pt;
    font-weight: 800;
    line-height: 1.25;
    color: #0f172a;
    text-align: center;
    margin-top: 5px;
    margin-bottom: 12px;
}}

.authors-meta {{
    text-align: center;
    font-size: 9pt;
    color: #475569;
    margin-bottom: 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid #e2e8f0;
}}

h2 {{
    font-size: 12.5pt;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 26px;
    margin-bottom: 10px;
    page-break-after: avoid;
    clear: both;
}}

h3 {{
    font-size: 10.5pt;
    font-weight: 700;
    color: #334155;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
}}

h4 {{
    font-size: 9.5pt;
    font-weight: 600;
    color: #475569;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
}}

p {{
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
    text-justify: inter-word;
}}

.table-container {{
    margin: 14px 0;
    page-break-inside: avoid;
    width: 100%;
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin: 0 auto;
}}

th, td {{
    padding: 5px 7px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}}

th {{
    background-color: #f8fafc;
    font-weight: 700;
    color: #0f172a;
    border-top: 2px solid #0f172a;
    border-bottom: 2px solid #0f172a;
}}

tr:last-child td {{
    border-bottom: 2px solid #0f172a;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 14px auto;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    page-break-inside: avoid;
}}

.mermaid-container {{
    display: block;
    width: 100%;
    margin: 18px 0;
    text-align: center;
    page-break-inside: avoid;
    clear: both;
}}

.mermaid {{
    display: inline-block;
    max-width: 90%;
    margin: 0 auto;
}}

.mermaid svg {{
    max-width: 100% !important;
    height: auto !important;
}}

.alert {{
    padding: 10px 14px;
    margin: 14px 0;
    border-left: 4px solid #3b82f6;
    background-color: #f0f9ff;
    font-size: 8.5pt;
    line-height: 1.5;
    border-radius: 3px;
    page-break-inside: avoid;
}}

.alert-warning {{
    border-left-color: #f59e0b;
    background-color: #fffbeb;
}}

.alert-important {{
    border-left-color: #ef4444;
    background-color: #fef2f2;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 18px 0;
}}

ul, ol {{
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 18px;
}}

li {{
    margin-bottom: 3px;
}}

code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8pt;
    background: #f1f5f9;
    padding: 1.5px 3.5px;
    border-radius: 3px;
    color: #991b1b;
}}

em {{
    color: #475569;
}}
</style>

<!-- MathJax 3 with SVG renderer for crisp PDF print -->
<script>
window.MathJax = {{
    tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
    }},
    svg: {{
        fontCache: 'global'
    }},
    options: {{
        enableMenu: false
    }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

<!-- Mermaid JS -->
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {{
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'default',
        flowchart: {{ useMaxWidth: true, htmlLabels: true }}
    }});
}});
</script>
</head>
<body>

{html_body}

</body>
</html>
"""
    return html_doc

def main():
    proj_dir = Path(__file__).resolve().parent.parent
    paper_md = proj_dir / "docs" / "topodlri" / "RESEARCH_PAPER.md"
    
    if not paper_md.exists():
        print(f"ERROR: {paper_md} not found!")
        sys.exit(1)

    print(f"Reading {paper_md}...")
    md_text = paper_md.read_text(encoding="utf-8")
    
    print("Converting Markdown to publication-grade HTML with MathJax SVG...")
    html_content = convert_markdown_to_html(md_text, paper_md.parent)
    
    temp_html = proj_dir / "docs" / "topodlri" / "paper_print.html"
    temp_html.write_text(html_content, encoding="utf-8")
    print(f"Saved intermediate HTML to {temp_html}")

    output_pdf = proj_dir / "docs" / "topodlri" / "RESEARCH_PAPER.pdf"
    output_pdf_root = proj_dir / "docs" / "RESEARCH_PAPER.pdf"

    print(f"Generating clean PDF with Google Chrome Headless to {output_pdf}...")
    cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={output_pdf}",
        f"file://{temp_html.resolve()}"
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"ERROR generating PDF: {res.stderr}")
        sys.exit(res.returncode)

    if output_pdf.exists():
        size_kb = output_pdf.stat().st_size / 1024
        print(f"SUCCESS: Generated {output_pdf} ({size_kb:.1f} KB)")
        output_pdf_root.write_bytes(output_pdf.read_bytes())
        print(f"Copied to {output_pdf_root}")
    else:
        print("ERROR: Output PDF was not created!")
        sys.exit(1)

if __name__ == "__main__":
    main()
