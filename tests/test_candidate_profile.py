import pytest

from app.services.candidate_profile import (
    CandidateProfileError,
    build_candidate_profile,
    extract_education_evidence,
    extract_resume_sections,
    extract_skills,
)


SAMPLE_RESUME = """
PROFESSIONAL SUMMARY

AI and Machine Learning graduate with experience in Python projects.

TARGET ROLES:

Junior Data Analyst, Junior Data Scientist.

TECHNICAL SKILLS:

Python, SQL, Pandas, NumPy, Scikit-learn, Git and GitHub.

WORK EXPERIENCE:

Completed practical data analysis and machine learning projects.

PROJECTS:

Built a Python project for data cleaning and exploratory data analysis.

EDUCATION:

B.Tech in Artificial Intelligence and Machine Learning.

CAREER PREFERENCES:

Entry-level full-time roles. Preferred locations: Mumbai and Pune.
"""


def test_extracts_explicit_skills():
    skills = extract_skills(
        "Python, SQL, Pandas, NumPy, Scikit-learn and PyTorch."
    )

    assert "Python" in skills
    assert "SQL" in skills
    assert "Pandas" in skills
    assert "NumPy" in skills
    assert "Scikit-learn" in skills
    assert "PyTorch" in skills


def test_does_not_invent_unmentioned_skills():
    skills = extract_skills(
        "Interested in software development and learning new technologies."
    )

    assert skills == []
    assert "Scala" not in skills
    assert "Kubernetes" not in skills


def test_identifies_resume_sections():
    sections = extract_resume_sections(SAMPLE_RESUME)

    assert "summary" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "projects" in sections
    assert "education" in sections
    assert "target_roles" in sections
    assert "preferences" in sections


def test_extracts_education_evidence():
    education = extract_education_evidence(SAMPLE_RESUME)

    assert len(education) == 1
    assert "B.Tech" in education[0]


def test_builds_profile_with_explicit_user_target_roles():
    profile = build_candidate_profile(
        SAMPLE_RESUME,
        target_roles=[
            "Junior Data Analyst",
            "",
            "Junior Data Scientist",
        ],
    )

    assert profile.target_roles == [
        "Junior Data Analyst",
        "Junior Data Scientist",
    ]

    assert "Python" in profile.skills
    assert "B.Tech" in profile.education_evidence[0]
    assert "data cleaning" in profile.matching_text.lower()


def test_excludes_target_roles_and_preferences_from_matching_text():
    profile = build_candidate_profile(SAMPLE_RESUME)

    assert "Junior Data Analyst" not in profile.matching_text
    assert "Mumbai" not in profile.matching_text
    assert "Pune" not in profile.matching_text
    assert "Python" in profile.matching_text


def test_does_not_infer_target_roles_from_resume():
    profile = build_candidate_profile(SAMPLE_RESUME)

    assert profile.target_roles == []


def test_rejects_empty_resume_text():
    with pytest.raises(CandidateProfileError):
        build_candidate_profile("   ")


def test_removes_cover_letter_template_from_resume():
    resume = """
    2026 B.Tech AI & ML graduate and AWS Certified Cloud Practitioner,
    and I write What draws me to [Company] is [one specific,
    verifiable fact: a product launch or system migration].
    That tells me the [system] where my skills apply.
    """

    profile = build_candidate_profile(resume)

    assert "What draws me to" not in profile.matching_text
    assert "[Company]" not in profile.matching_text
    assert "one specific" not in profile.matching_text
    # A degree mentioned in a summary is not formal education evidence.
    assert profile.education_evidence == []


def test_extracts_concise_education_evidence():
    resume = """
    EDUCATION

    2026 B.Tech AI & ML graduate and AWS Certified Cloud Practitioner.
    """

    education = extract_education_evidence(resume)

    assert len(education) == 1
    assert education[0] == "2026 B.Tech AI & ML graduate"


def test_removes_template_instructions_from_matching_text():
    resume = """
    TECHNICAL SKILLS
    Python, SQL, MySQL.

    PROJECTS
    Built data analysis projects.

    What draws me to [Company] is [one specific, verifiable fact].
    """

    profile = build_candidate_profile(resume)

    assert "Python" in profile.skills
    assert "SQL" in profile.skills
    assert "Built data analysis projects" in profile.matching_text
    assert "verifiable fact" not in profile.matching_text


def test_extracts_complete_education_entries_without_summary_contamination():
    resume = """
    PROFESSIONAL SUMMARY

    B.Tech AI & ML graduate (2026) targeting QA and test automation roles.

    EDUCATION

    B.Tech, AI & ML, Example Institute | 2022-2025 | CGPA 8.2
    Diploma, Computer Science, Sample Polytechnic | 2019-2022 | 75%
    """

    profile = build_candidate_profile(resume)

    assert profile.education_evidence == [
        "B.Tech, AI & ML, Example Institute | 2022-2025 | CGPA 8.2",
        "Diploma, Computer Science, Sample Polytechnic | 2019-2022 | 75%",
    ]

    assert all(
        "targeting QA" not in item
        for item in profile.education_evidence
    )


def test_does_not_infer_formal_education_from_summary():
    resume = """
    PROFESSIONAL SUMMARY

    B.Tech AI & ML graduate (2026) targeting QA roles.

    TECHNICAL SKILLS

    Python, SQL, Java.
    """

    profile = build_candidate_profile(resume)

    assert profile.education_evidence == []
