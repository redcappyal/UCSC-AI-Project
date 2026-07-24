"""Build docs/annotation-guide.pdf from annotation-guide.md.

    python3 docs/build_annotation_pdf.py

Reuses the markdown styling from build_spec_pdf.py so both project PDFs look
alike. Text-only, so unlike the mount spec this needs no rsvg-convert - just
reportlab.
"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).parent))
from build_spec_pdf import S, clean, inline  # noqa: E402  shared styling

DOCS = Path(__file__).parent
MD = DOCS / "annotation-guide.md"
OUT = DOCS / "annotation-guide.pdf"
TITLE = "CrossCourt - ball annotation guide"

# Checklist items read as boxes to tick on a printout, not as literal markdown.
CHECKBOX_RE = re.compile(r"^\[[ xX]\]\s*")


def render(lines):
    story, i = [], 0

    def flush_table(rows):
        ncols = max(len(r) for r in rows)
        data = []
        for ri, row in enumerate(rows):
            row = row + [""] * (ncols - len(row))
            style = S["cellh"] if ri == 0 else S["cell"]
            data.append([Paragraph(inline(c), style) for c in row])
        total = 6.9 * inch
        # A two-column table is a key/value list, so give the key column room;
        # wider tables split evenly.
        first = total * 0.34 if ncols == 2 else total / ncols
        rest = (total - first) / (ncols - 1) if ncols > 1 else total
        table = Table(data, colWidths=[first] + [rest] * (ncols - 1), repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B4B2A9")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1EFE8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.extend([Spacer(1, 4), table, Spacer(1, 4)])

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
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
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.fullmatch(r"[\s:\-|]+", lines[i].strip().strip("|")):
                    rows.append(cells)
                i += 1
            flush_table(rows)
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), S["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), S["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), S["h3"]))
        elif line.startswith("- "):
            item = line[2:]
            while (i + 1 < len(lines) and lines[i + 1].startswith("  ")
                   and not lines[i + 1].lstrip().startswith("- ")):
                i += 1
                item += " " + lines[i].strip()
            bullet = "•"
            if CHECKBOX_RE.match(item):
                # "[ ]" rather than a box glyph: U+2610 is absent from the
                # built-in fonts and renders as a solid black square.
                item, bullet = CHECKBOX_RE.sub("", item), "[  ]"
            story.append(Paragraph(inline(item), S["bullet"], bulletText=bullet))
        elif line.strip():
            para = line
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not re.match(r"^(#|\||- |```)", lines[i + 1])):
                i += 1
                para += " " + lines[i].strip()
            story.append(Paragraph(inline(para), S["body"]))
            story.append(Spacer(1, 3))
        i += 1
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#888780"))
    canvas.drawString(0.8 * inch, 0.5 * inch, TITLE)
    canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.7 * inch, bottomMargin=0.75 * inch, title=TITLE,
        subject="Labeling practices for the CrossCourt ball detector")
    doc.build(render(MD.read_text().splitlines()),
              onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)
