
from pathlib import Path
from xml.sax.saxutils import escape

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, Preformatted
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "PathSnap_Technical_Report.pdf"

SECTIONS = [
    "01_project_overview.md",
    "02_dataset_eda.md",
    "03_model_experimentation.md",
    "04_bayesian_optimisation.md",
    "05_hmm_route_reconstruction.md",
    "06_geographical_visualisation.md",
    "07_api_ui_deployment.md",
    "08_conclusion.md",
]

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    fontSize=25, leading=32, alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"), spaceAfter=15
))
styles.add(ParagraphStyle(
    name="SectionHeading", parent=styles["Heading1"],
    fontSize=17, leading=22, textColor=colors.HexColor("#17365D"),
    spaceBefore=12, spaceAfter=10
))
styles.add(ParagraphStyle(
    name="SubHeading", parent=styles["Heading2"],
    fontSize=13, leading=17, textColor=colors.HexColor("#24527A"),
    spaceBefore=10, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="SmallBody", parent=styles["BodyText"],
    fontSize=9, leading=13, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="CodeStyle", fontName="Courier",
    fontSize=7.5, leading=10, leftIndent=8,
    backColor=colors.HexColor("#F1F3F5"),
    borderPadding=6, spaceBefore=5, spaceAfter=8
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["Italic"],
    fontSize=8, leading=11, alignment=TA_CENTER,
    textColor=colors.grey, spaceAfter=8
))

story = []


def inline_html(node):
    """Convert basic HTML inline formatting to ReportLab markup."""
    if isinstance(node, NavigableString):
        return escape(str(node))

    if not isinstance(node, Tag):
        return ""

    content = "".join(inline_html(child) for child in node.children)
    tag = node.name.lower()

    if tag in ("strong", "b"):
        return f"<b>{content}</b>"
    if tag in ("em", "i"):
        return f"<i>{content}</i>"
    if tag == "code":
        return f'<font name="Courier">{content}</font>'
    if tag == "a":
        return content

    return content


def add_image(src, alt=""):
    path = (ROOT / src).resolve()

    if not path.exists():
        story.append(Paragraph(
            f"[Image not found: {escape(src)}]",
            styles["Caption"]
        ))
        return

    img = Image(str(path))
    max_width = 165 * mm
    max_height = 220 * mm

    scale = min(max_width / img.imageWidth,
                max_height / img.imageHeight, 1)

    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"

    story.append(Spacer(1, 5))
    story.append(img)

    if alt:
        story.append(Paragraph(escape(alt), styles["Caption"]))

    story.append(Spacer(1, 8))


def add_table(element):
    rows = []

    for tr in element.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            cells.append(Paragraph(
                inline_html(cell),
                styles["SmallBody"]
            ))
        if cells:
            rows.append(cells)

    if not rows:
        return

    col_count = max(len(row) for row in rows)
    for row in rows:
        while len(row) < col_count:
            row.append("")

    available_width = 165 * mm
    table = Table(
        rows,
        colWidths=[available_width / col_count] * col_count,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(Spacer(1, 6))
    story.append(table)
    story.append(Spacer(1, 10))


def process_element(element):
    if isinstance(element, NavigableString):
        return

    if not isinstance(element, Tag):
        return

    tag = element.name.lower()

    if tag in ("h1", "h2", "h3"):
        text = inline_html(element)
        style = (
            styles["SectionHeading"] if tag == "h1"
            else styles["SubHeading"]
        )
        story.append(Paragraph(text, style))

    elif tag == "p":
        images = element.find_all("img")
        if images:
            for img in images:
                add_image(img.get("src", ""), img.get("alt", ""))
        else:
            text = inline_html(element)
            if text.strip():
                story.append(Paragraph(text, styles["SmallBody"]))

    elif tag in ("ul", "ol"):
        for li in element.find_all("li", recursive=False):
            text = inline_html(li)
            story.append(Paragraph(
                "• " + text,
                styles["SmallBody"]
            ))
        story.append(Spacer(1, 4))

    elif tag == "table":
        add_table(element)

    elif tag == "pre":
        code = element.get_text()
        story.append(Preformatted(code, styles["CodeStyle"]))

    elif tag == "blockquote":
        story.append(Paragraph(
            inline_html(element),
            styles["SmallBody"]
        ))

    elif tag == "hr":
        story.append(Spacer(1, 8))

    else:
        for child in element.children:
            process_element(child)


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)

    canvas.drawString(
        22 * mm, 12 * mm,
        "PathSnap - Technical Report"
    )
    canvas.drawRightString(
        188 * mm, 12 * mm,
        f"Page {doc.page}"
    )
    canvas.restoreState()


def main():
    missing = [f for f in SECTIONS if not (ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing report sections: " + ", ".join(missing)
        )

    # Cover page
    story.append(Spacer(1, 45 * mm))
    story.append(Paragraph(
        "PathSnap",
        styles["ReportTitle"]
    ))
    story.append(Paragraph(
        "GPS Map Matching and Route Reconstruction",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph(
        "Consolidated Technical Report",
        ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Normal"],
            fontSize=15,
            alignment=TA_CENTER
        )
    ))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph(
        "UE24CS352A",
        ParagraphStyle(
            "Course",
            parent=styles["Normal"],
            fontSize=11,
            alignment=TA_CENTER
        )
    ))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "November 2026",
        ParagraphStyle(
            "Date",
            parent=styles["Normal"],
            fontSize=10,
            alignment=TA_CENTER
        )
    ))
    story.append(PageBreak())

    # Contents
    story.append(Paragraph("Contents", styles["SectionHeading"]))
    contents_titles = [
        "Project Overview",
        "Dataset EDA",
        "Model Experimentation",
        "Bayesian Optimisation",
        "HMM Route Reconstruction",
        "Geographical Visualisation",
        "API, UI and Deployment",
        "Conclusion",
    ]

    for index, filename in enumerate( SECTIONS, 1):
        title = contents_titles[index - 1]
        story.append(Paragraph(
            f"{index}. {escape(title)}",
            styles["SmallBody"]
        ))

    story.append(PageBreak())

    # Report sections
    for index, filename in enumerate(SECTIONS):
        path = ROOT / filename
        text = path.read_text(encoding="utf-8")

        html = markdown.markdown(
            text,
            extensions=["tables", "fenced_code", "sane_lists"]
        )

        soup = BeautifulSoup(html, "html.parser")

        for element in soup.children:
            process_element(element)

        if index < len(SECTIONS) - 1:
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        title="PathSnap Technical Report",
        author="PathSnap Project Team"
    )

    doc.build(
        story,
        onFirstPage=page_number,
        onLaterPages=page_number
    )

    print(f"\nPDF created successfully: {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
