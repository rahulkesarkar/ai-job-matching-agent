import pymupdf
import pytest

from app.services.resume_pipeline import process_resume_pdf
from app.services.resume_parser import ResumeParsingError


def create_resume_pdf():
    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (72, 72),
        """
        PROFESSIONAL SUMMARY
        AI and Machine Learning graduate.

        TECHNICAL SKILLS
        Python, SQL, Pandas, NumPy, Scikit-learn.

        EDUCATION
        B.Tech in Artificial Intelligence and Machine Learning.

        PROJECTS
        Built machine learning projects using Python.
        """
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_process_resume_pdf_end_to_end():
    result = process_resume_pdf(
        create_resume_pdf(),
        target_roles=["Junior Data Analyst", "Junior ML Engineer"],
    )

    assert result.extraction.page_count == 1
    assert result.extraction.extracted_characters > 0

    assert "Python" in result.profile.skills
    assert "SQL" in result.profile.skills
    assert "Pandas" in result.profile.skills

    assert result.profile.target_roles == [
        "Junior Data Analyst",
        "Junior ML Engineer",
    ]

    assert "B.Tech" in result.profile.education_evidence[0]


def test_target_roles_are_not_inferred():
    result = process_resume_pdf(create_resume_pdf())

    assert result.profile.target_roles == []


def test_invalid_pdf_is_rejected():
    with pytest.raises(ResumeParsingError):
        process_resume_pdf(b"not a valid PDF")
