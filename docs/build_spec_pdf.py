"""Build docs/mount-spec.pdf: spec pages from mount-spec.md + the three
orthographic view figures (docs/figures/*.svg) as vector landscape pages.

    python3 docs/build_spec_pdf.py

Needs: reportlab, pypdf (pip) and rsvg-convert on PATH (brew install librsvg).
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from pypdf import PdfReader, PdfWriter

DOCS = Path(__file__).parent
MD = DOCS / "mount-spec.md"
OUT = DOCS / "mount-spec.pdf"
FIGURES = [("view1_side.svg", 560), ("view2_front.svg", 545), ("view3_plan.svg", 545)]

REPL = {"⇒": "=>", "≤": "<=", "≥": ">=", "✓": "[ok]",
        "←": "<-", "→": "->", "≈": "~", "✕": "x", "✖": "x", "✗": "x"}


def clean(s):
    for k, v in REPL.items():
        s = s.replace(k, v)
    return s


def inline(s):
    s = clean(s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.5">\1</font>', s)
    return s


styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=styles["Title"], fontSize=16, leading=20,
                            spaceAfter=6, alignment=0),
    "h2": ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, leading=15,
                         spaceBefore=12, spaceAfter=4,
                         textColor=colors.HexColor("#26215C")),
    "h3": ParagraphStyle("h3", parent=styles["Heading3"], fontSize=10.5, leading=13,
                         spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("b", parent=styles["Normal"], fontSize=9.5, leading=13),
    "bullet": ParagraphStyle("bl", parent=styles["Normal"], fontSize=9.5, leading=13,
                             leftIndent=14, bulletIndent=4, spaceAfter=2),
    "cell": ParagraphStyle("c", parent=styles["Normal"], fontSize=8.5, leading=11),
    "cellh": ParagraphStyle("ch", parent=styles["Normal"], fontSize=8.5, leading=11,
                            fontName="Helvetica-Bold"),
}


def build_spec_pages(tmp):
    lines = MD.read_text().splitlines()
    story, i = [], 0

    def flush_table(rows):
        ncols = max(len(r) for r in rows)
        data = []
        for ri, r in enumerate(rows):
            r = r + [""] * (ncols - len(r))
            st = S["cellh"] if ri == 0 else S["cell"]
            data.append([Paragraph(inline(c), st) for c in r])
        total = 6.9 * inch
        w0 = total * (1.6 if ncols > 2 else 1.2) / (ncols + (0.6 if ncols > 2 else 0.2))
        rest = (total - w0) / (ncols - 1) if ncols > 1 else total
        t = Table(data, colWidths=[w0] + [rest] * (ncols - 1), repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B4B2A9")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1EFE8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.extend([Spacer(1, 4), t, Spacer(1, 4)])

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(clean(lines[i]))
                i += 1
            i += 1
            story.append(Preformatted("\n".join(block), ParagraphStyle(
                "pre", fontName="Courier", fontSize=8, leading=10.5,
                leftIndent=10, spaceBefore=4, spaceAfter=4)))
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.fullmatch(r"[\s:\-|]+", lines[i].strip().strip("|")):
                    rows.append(cells)
                i += 1
            flush_table(rows)
            continue
        if ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), S["title"]))
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), S["h2"]))
        elif ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), S["h3"]))
        elif ln.startswith("- "):
            item = ln[2:]
            while (i + 1 < len(lines) and lines[i + 1].startswith("  ")
                   and not lines[i + 1].lstrip().startswith("- ")):
                i += 1
                item += " " + lines[i].strip()
            story.append(Paragraph(inline(item), S["bullet"], bulletText="•"))
        elif ln.strip():
            para = ln
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not re.match(r"^(#|\||- |```)", lines[i + 1])):
                i += 1
                para += " " + lines[i].strip()
            story.append(Paragraph(inline(para), S["body"]))
            story.append(Spacer(1, 3))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#888780"))
        canvas.drawString(0.8 * inch, 0.5 * inch,
                          "Fin mount - 3D modeling specification")
        canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch,
                               f"page {doc.page}")
        canvas.restoreState()

    spec = tmp / "spec.pdf"
    doc = SimpleDocTemplate(str(spec), pagesize=letter,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="Fin mount - 3D modeling specification")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return spec


def convert_figures(tmp):
    if not shutil.which("rsvg-convert"):
        raise SystemExit("rsvg-convert not found - install librsvg "
                         "(macOS: brew install librsvg)")
    pages = []
    for name, h in FIGURES:
        # 680 px = 510 pt natural; zoom 1.37 -> ~699 pt on a landscape-letter page
        ch = h * 0.75 * 1.37
        top = (612 - ch) / 2
        out = tmp / (name + ".pdf")
        subprocess.run(["rsvg-convert", "-f", "pdf", "--page-width", "792pt",
                        "--page-height", "612pt", "-z", "1.37", "--left", "46pt",
                        "--top", f"{top:.1f}pt", "-o", str(out),
                        str(DOCS / "figures" / name)], check=True)
        pages.append(out)
    return pages


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parts = [build_spec_pages(tmp)] + convert_figures(tmp)
        writer = PdfWriter()
        for part in parts:
            for page in PdfReader(str(part)).pages:
                writer.add_page(page)
        writer.add_metadata({
            "/Title": "Fin mount - 3D modeling specification",
            "/Subject": "Squash court fin mount: spec + three orthographic views"})
        with open(OUT, "wb") as f:
            writer.write(f)
    print("wrote", OUT, "pages:", len(PdfReader(str(OUT)).pages))
