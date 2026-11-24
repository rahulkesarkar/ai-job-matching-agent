import json

import httpx
import pytest

from app.services.candidate_profile import CandidateProfile
from app.services.llm_job_analyzer import analyze_job


@pytest.fixture
def candidate_profile():
    return CandidateProfile(
        skills=["Python", "SQL"],
        education_evidence=[
            "B.Tech, AI & ML, Example Institute | 2022-2025"
        ],
        sections={
            "skills": "Python, SQL, candidate@example.com",
            "projects": (
                "Built a Python data analysis project using SQL."
            ),
        },
        target_roles=["Junior Data Analyst"],
        matching_text="Python SQL data analysis",
    )


@pytest.fixture
def sample_job():
    return {
        "id": "job_001",
        "title": "Junior Data Analyst",
        "company": "Example Company",
        "location": "Mumbai, India",
        "description": (
            "Requirements: Python, SQL, Power BI and Tableau. "
            "Build reports and analyze business data."
        ),
    }


class FakeOllamaResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {
                "content": json.dumps(self.content)
            }
        }


def test_computes_skill_matches_and_gaps_deterministically(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def ollama_unavailable(*args, **kwargs):
        raise httpx.ConnectError("Ollama unavailable")

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        ollama_unavailable,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == ["Power BI", "Tableau"]

    assert result.used_fallback is True
    assert result.provider == "deterministic_fallback"

    assert "Python" in result.match_summary
    assert "Power BI" in result.recommendation


def test_uses_valid_local_llm_response(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]

        return FakeOllamaResponse({
            "match_summary": (
                "The candidate demonstrates Python and SQL, "
                "which are listed in the job requirements."
            ),
            "recommendation": (
                "Consider developing Power BI and Tableau skills."
            ),
        })

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        fake_post,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.provider == "ollama"
    assert result.used_fallback is False
    assert result.model == "qwen3:4b"

    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == ["Power BI", "Tableau"]

    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert captured["payload"]["keep_alive"] == 0

    # Contact details must not be sent to the local model.
    prompt = json.dumps(captured["payload"])
    assert "candidate@example.com" not in prompt
    assert "[redacted email]" in prompt


def test_rejects_unsupported_skill_claims_from_llm(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def hallucinated_response(url, **kwargs):
        return FakeOllamaResponse({
            "match_summary": (
                "The candidate demonstrates Python and Tableau."
            ),
            "recommendation": (
                "Use Tableau experience to improve reporting."
            ),
        })

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        hallucinated_response,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.used_fallback is True
    assert result.provider == "deterministic_fallback"

    assert "Tableau" not in result.matched_skills
    assert result.skill_gaps == ["Power BI", "Tableau"]


def test_falls_back_when_llm_returns_invalid_json(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def invalid_response(url, **kwargs):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "message": {
                        "content": "not valid JSON"
                    }
                }

        return Response()

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        invalid_response,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.used_fallback is True
    assert result.provider == "deterministic_fallback"
    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == ["Power BI", "Tableau"]


def test_rejects_job_without_description(candidate_profile):
    job = {
        "id": "job_002",
        "title": "Data Analyst",
        "company": "Example Company",
        "location": "Mumbai, India",
        "description": "",
    }

    with pytest.raises(ValueError, match="no description"):
        analyze_job(candidate_profile, job)


def test_rejects_job_without_id(candidate_profile):
    job = {
        "title": "Data Analyst",
        "company": "Example Company",
        "location": "Mumbai, India",
        "description": "Python and SQL required.",
    }

    with pytest.raises(ValueError, match="must have an ID"):
        analyze_job(candidate_profile, job)


def test_allows_llm_to_explain_missing_skills(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def valid_negative_response(url, **kwargs):
        return FakeOllamaResponse({
            "match_summary": (
                "The candidate demonstrates Python and SQL. "
                "Tableau is not demonstrated in the resume."
            ),
            "recommendation": (
                "The candidate should consider learning Power BI "
                "and Tableau if these tools are relevant to the role."
            ),
        })

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        valid_negative_response,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.used_fallback is False
    assert result.provider == "ollama"
    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == ["Power BI", "Tableau"]


def test_ignores_incidental_skills_in_company_description(
    monkeypatch,
    candidate_profile,
):
    from app.services.llm_job_analyzer import _extract_job_skill_requirements

    description = """
    ABOUT THE COMPANY

    Our team uses Python, Java, Tableau and AWS across our platform.

    REQUIREMENTS

    Python and SQL are required.

    PREFERRED QUALIFICATIONS

    Power BI and Tableau are nice to have.
    """

    required, preferred = _extract_job_skill_requirements(description)

    assert required == ["Python", "SQL"]
    assert preferred == ["Power BI", "Tableau"]


def test_preserves_education_dates_and_redacts_actual_phone_numbers():
    from app.services.llm_job_analyzer import _safe_text

    education = (
        "B.Tech, AI & ML, Example Institute "
        "| 2022-2025 | CGPA 8.2"
    )

    assert _safe_text(education) == education

    text = "Contact: 9876543210"
    assert "[redacted phone]" in _safe_text(text)


def test_preferred_skills_are_not_counted_as_required_gaps(
    monkeypatch,
    candidate_profile,
):
    from app.services.llm_job_analyzer import analyze_job

    description = """
    REQUIREMENTS

    Python and SQL are required.

    PREFERRED QUALIFICATIONS

    Power BI and Tableau are nice to have.
    """

    def unavailable(*args, **kwargs):
        import httpx
        raise httpx.ConnectError("Offline")

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        unavailable,
    )

    result = analyze_job(
        candidate_profile,
        {
            "id": "job_preferred_001",
            "title": "Junior Data Analyst",
            "company": "Example",
            "location": "Mumbai",
            "description": description,
        },
    )

    assert result.required_skills == ["Python", "SQL"]
    assert result.preferred_skills == ["Power BI", "Tableau"]
    assert result.matched_skills == ["Python", "SQL"]
    assert result.skill_gaps == []
    assert result.preferred_skill_gaps == ["Power BI", "Tableau"]


def test_extracts_requirements_from_mantle_skills_required_format():
    from app.services.llm_job_analyzer import (
        _extract_job_skill_requirements,
    )

    description = """
    ABOUT THE COMPANY

    We use Python, Java, Tableau and AWS across our platform.

    Experience: 1 - 3 Years

    Duties & Responsibilities:

    Develop dashboards using Excel, Power BI/Tableau and SQL.
    Analyze business data and report campaign performance.

    Skills Required

    Data analysis concepts and reporting methodologies

    Data visualization best practices

    Knowledge in SQL, MS Office, Power BI, Looker studio, Tableau.

    Strong data analyzing, cleaning and validation skills
    """

    required, preferred = _extract_job_skill_requirements(description)

    assert {
        "Data Analysis",
        "Data Visualization",
        "SQL",
        "Power BI",
        "Tableau",
    }.issubset(set(required))

    assert "Python" not in required
    assert "AWS" not in required
    assert "Excel" not in required
    assert preferred == []


def test_fallback_does_not_return_empty_required_skill_gap():
    from app.services.llm_job_analyzer import _deterministic_narrative

    summary, recommendation = _deterministic_narrative(
        matched_skills=[],
        skill_gaps=[],
        preferred_skill_gaps=[],
        required_skills=["Python"],
    )

    assert "did not return a complete result" in summary
    assert "manually" in recommendation.lower()
    assert "not detected in the resume: ." not in recommendation


def test_python_controls_recommendation_when_llm_corrupts_skill_names(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def corrupted_response(url, **kwargs):
        return FakeOllamaResponse({
            "match_summary": (
                "The candidate matches Python and SQL. "
                "Power BI and Tableau are not detected in the resume."
            ),
            "recommendation": (
                "Develop Power and Table skills, along with Tableau."
            ),
        })

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        corrupted_response,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.used_fallback is False
    assert result.provider == "ollama"
    assert "Power BI" in result.recommendation
    assert "Tableau" in result.recommendation
    assert "Power and Table" not in result.recommendation


def test_python_includes_all_gaps_when_llm_omits_a_skill(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def incomplete_response(url, **kwargs):
        return FakeOllamaResponse({
            "match_summary": (
                "The candidate matches Python and SQL. "
                "Power BI and Tableau are not detected in the resume."
            ),
            "recommendation": (
                "Develop Power BI skills to improve reporting."
            ),
        })

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        incomplete_response,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.used_fallback is False
    assert result.provider == "ollama"
    assert "Power BI" in result.recommendation
    assert "Tableau" in result.recommendation




def test_candidate_evidence_balances_resume_sections():
    from app.services.llm_job_analyzer import _candidate_evidence

    profile = CandidateProfile(
        skills=["Python", "SQL", "Power BI"],
        education_evidence=["B.Tech in Artificial Intelligence"],
        sections={
            "skills": "Python, SQL\nPower BI\nExcel\nGit",
            "experience": (
                "Developed automated QA workflows using Python and SQL.\n"
                "Maintained test automation and CI workflows."
            ),
            "projects": (
                "Built a data analysis project using Python and Pandas.\n"
                "Created a reporting dashboard using SQL and Power BI."
            ),
            "certifications": "AWS Cloud Practitioner certification",
        },
        target_roles=["Junior Data Analyst"],
        matching_text="Python SQL Power BI",
    )

    evidence = _candidate_evidence(profile)

    assert any(item.startswith("Skills:") for item in evidence)
    assert any(item.startswith("Experience:") for item in evidence)
    assert any(item.startswith("Projects:") for item in evidence)
    assert any(item.startswith("Certifications:") for item in evidence)
    assert any(item.startswith("Education:") for item in evidence)
    assert len(evidence) <= 7


def test_required_skill_coverage_is_calculated_deterministically(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def ollama_unavailable(*args, **kwargs):
        raise httpx.ConnectError("Ollama unavailable")

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        ollama_unavailable,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.required_skill_match_count == 2
    assert result.required_skill_count == 4
    assert result.required_skill_coverage == 0.5


def test_job_aware_evidence_prioritizes_relevant_dashboard_project():
    from app.services.llm_job_analyzer import _candidate_evidence

    profile = CandidateProfile(
        skills=["Python", "SQL", "Power BI", "Selenium"],
        education_evidence=[],
        sections={
            "projects": (
                "Test Automation Framework | Selenium, Pytest, Python.\n"
                "E-Commerce Sales Dashboard | Excel, SQL, Power BI."
            ),
        },
        target_roles=["Junior Data Analyst"],
        matching_text="Python SQL Power BI Selenium",
    )

    evidence = _candidate_evidence(
        profile,
        relevant_skills={"SQL", "Power BI", "Data Analysis"},
    )

    project_evidence = [
        item for item in evidence if item.startswith("Projects:")
    ]

    assert len(project_evidence) == 2
    assert "E-Commerce Sales Dashboard" in project_evidence[0]



def test_rejects_uncalibrated_strong_alignment_claim(
    monkeypatch,
    candidate_profile,
    sample_job,
):
    def overconfident_response(url, **kwargs):
        return FakeOllamaResponse({
            "match_summary": (
                "The candidate has strong alignment with the "
                "Python and SQL requirements."
            ),
            "recommendation": (
                "Consider developing Power BI and Tableau skills."
            ),
        })

    monkeypatch.setattr(
        "app.services.llm_job_analyzer.httpx.post",
        overconfident_response,
    )

    result = analyze_job(candidate_profile, sample_job)

    assert result.used_fallback is True
    assert result.provider == "deterministic_fallback"
    assert "strong alignment" not in result.match_summary.lower()
    assert "2/4 (50%)" in result.match_summary
