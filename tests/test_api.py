from types import SimpleNamespace
import pandas as pd
import pymupdf
import pytest

from fastapi.testclient import TestClient

import app.main as api_module

from app.services.job_matching_engine import JobMatchResults
from app.services.llm_job_analyzer import JobAnalysisResult


class FakeMatchingEngine:
    """Lightweight engine for testing API behavior."""

    def __init__(self):
        # Mirror the production engine interface used by /analyze-job.
        self.eligible_jobs = pd.DataFrame()

    def match(self, profile):
        assert profile.matching_text

        sample_results = pd.DataFrame([
            {
                "title": "Junior Data Analyst",
                "company": "Example Company",
                "tfidf_similarity": 0.85,
            }
        ])

        semantic_results = pd.DataFrame([
            {
                "title": "Junior Data Scientist",
                "company": "Example Company",
                "semantic_similarity": 0.82,
            }
        ])

        diversified_results = pd.DataFrame([
            {
                "title": "Junior Data Scientist",
                "company": "Example Company",
                "semantic_similarity": 0.82,
                "duplicate_posting_count": 1,
            }
        ])

        return JobMatchResults(
            tfidf_ranked=sample_results,
            semantic_ranked=semantic_results,
            diversified_ranked=diversified_results,
            candidate_chunk_count=1,
            indexed_chunk_count=10095,
            model_device="cpu",
            eligible_job_count=7,
            excluded_job_count=42,
        )


@pytest.fixture
def client(monkeypatch):
    # Prevent tests from loading the real Sentence Transformer and FAISS index.
    monkeypatch.setattr(
        api_module,
        "JobMatchingEngine",
        FakeMatchingEngine,
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def create_test_pdf():
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
        Built data analysis projects using Python.
        """
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["matching_engine_ready"] is True


def test_upload_resume_endpoint(client):
    response = client.post(
        "/upload-resume",
        files={
            "file": (
                "sample_resume.pdf",
                create_test_pdf(),
                "application/pdf",
            )
        },
        data={
            "target_roles": "Junior Data Analyst",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "sample_resume.pdf"
    assert data["page_count"] == 1
    assert "Python" in data["skills"]
    assert "B.Tech in Artificial Intelligence and Machine Learning." in (
        data["education_evidence"]
    )
    assert data["target_roles"] == ["Junior Data Analyst"]
    assert data["profile_ready"] is True


def test_match_endpoint(client):
    response = client.post(
        "/match",
        files={
            "file": (
                "sample_resume.pdf",
                create_test_pdf(),
                "application/pdf",
            )
        },
        data={
            "target_roles": "Junior Data Analyst",
            "top_k": "5",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "Python" in data["candidate_skills"]
    assert data["target_roles"] == ["Junior Data Analyst"]
    assert data["candidate_pages"] == 1
    assert data["candidate_chunks"] == 1
    assert data["eligible_job_count"] == 7
    assert data["excluded_job_count"] == 42

    assert len(data["results"]["tfidf"]) == 1
    assert len(data["results"]["semantic"]) == 1
    assert len(data["results"]["diversified"]) == 1

    assert data["results"]["tfidf"][0]["company"] == (
        "Example Company"
    )


def test_invalid_pdf_returns_bad_request(client):
    response = client.post(
        "/upload-resume",
        files={
            "file": (
                "invalid.pdf",
                b"this is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_invalid_top_k_returns_validation_error(client):
    response = client.post(
        "/match",
        files={
            "file": (
                "sample_resume.pdf",
                create_test_pdf(),
                "application/pdf",
            )
        },
        data={"top_k": "0"},
    )

    assert response.status_code == 422


def test_match_endpoint_accepts_multiple_target_roles(client):
    response = client.post(
        "/match",
        files=[
            (
                "file",
                (
                    "sample_resume.pdf",
                    create_test_pdf(),
                    "application/pdf",
                ),
            ),
            ("target_roles", (None, "Junior Data Analyst")),
            ("target_roles", (None, "Junior Data Scientist")),
        ],
        data={"top_k": "5"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["target_roles"] == [
        "Junior Data Analyst",
        "Junior Data Scientist",
    ]


def test_analyze_job_endpoint_returns_validated_analysis(
    client,
    monkeypatch,
):
    eligible_jobs = pd.DataFrame([
        {
            "id": "job_001",
            "title": "Junior Data Analyst",
            "company": "Example Company",
            "location": "Mumbai, India",
            "description": "Python and SQL are required.",
        }
    ])

    def fake_filter(**kwargs):
        assert kwargs["target_roles"] == ["Junior Data Analyst"]
        return SimpleNamespace(eligible_jobs=eligible_jobs)

    def fake_analyze(profile, job):
        assert profile.matching_text
        assert job["id"] == "job_001"

        return JobAnalysisResult(
            job_id="job_001",
            job_title="Junior Data Analyst",
            company="Example Company",
            location="Mumbai, India",
            match_summary="Python and SQL are present in the resume.",
            required_skills=["Python", "SQL"],
            preferred_skills=[],
            matched_skills=["Python", "SQL"],
            skill_gaps=[],
            preferred_skill_gaps=[],
            required_skill_match_count=2,
            required_skill_count=2,
            required_skill_coverage=1.0,
            candidate_evidence=["Python and SQL projects."],
            job_evidence=["Python and SQL are required."],
            recommendation="Review the complete job description.",
            provider="deterministic_fallback",
            model="none",
            used_fallback=True,
        )

    monkeypatch.setattr(
        api_module,
        "filter_jobs_for_candidate",
        fake_filter,
    )
    monkeypatch.setattr(api_module, "analyze_job", fake_analyze)

    response = client.post(
        "/analyze-job",
        files=[
            (
                "file",
                (
                    "sample_resume.pdf",
                    create_test_pdf(),
                    "application/pdf",
                ),
            ),
            ("job_id", (None, "job_001")),
            ("target_roles", (None, "Junior Data Analyst")),
        ],
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["job_id"] == "job_001"
    assert data["matched_skills"] == ["Python", "SQL"]
    assert data["skill_gaps"] == []
    assert data["provider"] == "deterministic_fallback"
    assert data["used_fallback"] is True


def test_analyze_job_endpoint_rejects_ineligible_job(
    client,
    monkeypatch,
):
    eligible_jobs = pd.DataFrame([
        {
            "id": "job_001",
            "title": "Junior Data Analyst",
            "company": "Example Company",
            "location": "Mumbai, India",
            "description": "Python and SQL are required.",
        }
    ])

    monkeypatch.setattr(
        api_module,
        "filter_jobs_for_candidate",
        lambda **kwargs: SimpleNamespace(
            eligible_jobs=eligible_jobs
        ),
    )

    def should_not_analyze(*args, **kwargs):
        pytest.fail("An ineligible job must not reach the analyzer.")

    monkeypatch.setattr(
        api_module,
        "analyze_job",
        should_not_analyze,
    )

    response = client.post(
        "/analyze-job",
        files=[
            (
                "file",
                (
                    "sample_resume.pdf",
                    create_test_pdf(),
                    "application/pdf",
                ),
            ),
            ("job_id", (None, "job_rejected")),
            ("target_roles", (None, "Junior Data Analyst")),
        ],
    )

    assert response.status_code == 404
    assert "not eligible" in response.json()["detail"]
