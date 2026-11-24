import pymupdf
import pytest

from app.services.resume_parser import (
    ResumeParsingError,
    extract_resume_text,
)


def create_pdf(text=None, pages=1):
    """Generate a temporary PDF in memory."""
    document = pymupdf.open()

    for _ in range(pages):
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)

    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_text_from_valid_pdf():
    pdf_bytes = create_pdf(
        "Candidate: Sample Applicant\nSkills: Python, SQL"
    )

    result = extract_resume_text(pdf_bytes)

    assert result.page_count == 1
    assert "Python" in result.text
    assert "SQL" in result.text
    assert result.extracted_characters > 0


def test_reject_invalid_pdf():
    with pytest.raises(ResumeParsingError, match="not a valid PDF"):
        extract_resume_text(b"not a PDF")


def test_reject_empty_pdf():
    pdf_bytes = create_pdf()

    with pytest.raises(ResumeParsingError, match="No selectable text"):
        extract_resume_text(pdf_bytes)


def test_reject_empty_upload():
    with pytest.raises(ResumeParsingError, match="file is empty"):
        extract_resume_text(b"")


def test_reject_oversized_pdf():
    pdf_bytes = create_pdf("Sample resume")

    with pytest.raises(ResumeParsingError, match="size limit"):
        extract_resume_text(pdf_bytes, max_file_size_mb=0)


def test_reject_pdf_exceeding_page_limit():
    pdf_bytes = create_pdf("Sample resume", pages=3)

    with pytest.raises(ResumeParsingError, match="page limit"):
        extract_resume_text(pdf_bytes, max_pages=2)
