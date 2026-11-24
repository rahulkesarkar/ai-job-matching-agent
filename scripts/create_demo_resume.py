"""Generate a one-page synthetic PDF resume for local product demonstrations.

The generated file contains fictional candidate information only. It must never
be mistaken for a real person's resume or used as a real job application.
"""

from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data" / "sample" / "demo_candidate_resume.txt"
OUTPUT = PROJECT_ROOT / "data" / "sample" / "demo_candidate_resume.pdf"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8").strip()

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)  # A4 points

    margin = 48
    content = pymupdf.Rect(
        margin,
        margin,
        page.rect.width - margin,
        page.rect.height - margin,
    )

    remaining_space = page.insert_textbox(
        content,
        text,
        fontname="helv",
        fontsize=9.2,
        lineheight=1.25,
        color=(0.12, 0.16, 0.15),
        align=pymupdf.TEXT_ALIGN_LEFT,
    )

    # Do not write an incomplete PDF if the content does not fit.
    if remaining_space < 0:
        document.close()
        raise ValueError("Synthetic resume text does not fit on one A4 page.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT, garbage=4, deflate=True)
    document.close()

    # Re-open the output and verify that it is a readable, single-page PDF.
    with pymupdf.open(OUTPUT) as check:
        assert check.page_count == 1, "Demo resume must be one page."
        extracted = check[0].get_text()
        assert "Aarav Mehta" in extracted
        assert "SYNTHETIC DEMONSTRATION RESUME" in extracted

    print(f"Created and verified: {OUTPUT}")


if __name__ == "__main__":
    main()
