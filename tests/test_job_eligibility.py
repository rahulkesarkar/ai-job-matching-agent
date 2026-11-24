import pandas as pd
import pytest

from app.services.job_eligibility import (
    JobEligibilityError,
    extract_minimum_required_experience,
    extract_required_advanced_degree,
    filter_jobs_for_candidate,
    identify_target_categories,
    is_senior_job,
    match_target_role,
)


TARGET_ROLES = [
    "Junior Data Analyst",
    "Junior Data Scientist",
    "Entry-Level Machine Learning Engineer",
]


def make_job(
    job_id,
    title,
    description="Python and SQL skills.",
    country="India",
    seniority="Not Applicable",
):
    return {
        "id": job_id,
        "title": title,
        "company": "Test Company",
        "location": "Mumbai, Maharashtra, India",
        "description": description,
        "seniority_level": seniority,
        "normalized_country": country,
    }


def test_identifies_selected_role_categories():
    assert identify_target_categories(TARGET_ROLES) == [
        "data_analyst",
        "data_scientist",
        "ml_engineer",
    ]


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Junior Data Analyst", "data_analyst"),
        ("Associate Data Scientist", "data_scientist"),
        ("Machine Learning Engineer", "ml_engineer"),
        ("Junior AI & Data Engineer", "ml_engineer"),
        ("Associate AI Engineer", "ml_engineer"),
        ("Cloud Engineer", None),
        ("Manager Customer Support Engineering", None),
    ],
)
def test_matches_target_job_titles(title, expected):
    assert match_target_role(
        title,
        ["data_analyst", "data_scientist", "ml_engineer"],
    ) == expected


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Experience: 2-4 years in software development.", 2.0),
        ("Requires 5+ years of data engineering experience.", 5.0),
        ("Experience: 1 to 2 years in Python.", 1.0),
        ("Typically 18+ years of engineering experience.", 18.0),
        ("Freshers are welcome. No prior experience required.", None),
        ("Python and SQL are required.", None),
    ],
)
def test_extracts_explicit_experience_requirements(description, expected):
    assert extract_minimum_required_experience(description) == expected


@pytest.mark.parametrize(
    "description",
    [
        "With over 100 years of experience in financial services.",
        "More than 100 years of experience in the industry.",
        "2000+ years of combined experience in mobile technologies.",
        "Drawing on 150 years of experience in the rail sector.",
        "A heritage of 75 years of experience in manufacturing.",
    ],
)
def test_ignores_company_history_experience(description):
    assert extract_minimum_required_experience(description) is None


def test_detects_mandatory_phd():
    assert extract_required_advanced_degree(
        "PhD is mandatory in Computer Science or Statistics."
    ) == "PhD"


def test_detects_research_masters_requirement():
    text = (
        "A Master of Science degree in Data Science "
        "with a thesis or research projects involving machine learning."
    )

    assert extract_required_advanced_degree(text) == "Master's degree"


def test_does_not_reject_preferred_masters():
    assert extract_required_advanced_degree(
        "Bachelor's degree required. Master's degree preferred."
    ) is None


def test_does_not_reject_bachelors_requirement():
    assert extract_required_advanced_degree(
        "Bachelor's degree in Computer Science or equivalent."
    ) is None


def test_detects_seniority_and_early_career_override():
    assert is_senior_job(
        "Senior Director, AI Engineering",
        "Not Applicable",
    )

    assert is_senior_job(
        "Data Scientist",
        "Mid-Senior level",
    )

    # The description explicitly welcomes freshers despite bad metadata.
    assert not is_senior_job(
        "Junior AI & Data Engineer",
        "Mid-Senior level",
        "We are hiring smart analytical freshers and early-career engineers.",
    )

    assert not is_senior_job(
        "Junior Data Scientist",
        "Not Applicable",
    )


def test_keeps_suitable_entry_level_jobs():
    jobs = pd.DataFrame([
        make_job("1", "Junior Data Analyst"),
        make_job("2", "Junior Data Scientist"),
        make_job("3", "Machine Learning Engineer"),
    ])

    result = filter_jobs_for_candidate(jobs, TARGET_ROLES)

    assert result.eligible_count == 3
    assert result.excluded_count == 0


def test_keeps_early_career_job_with_inaccurate_seniority_metadata():
    jobs = pd.DataFrame([
        make_job(
            "omni",
            "Junior AI & Data Engineer",
            (
                "We are hiring smart analytical freshers and "
                "early-career engineers. Freshers are welcome."
            ),
            seniority="Mid-Senior level",
        )
    ])

    result = filter_jobs_for_candidate(jobs, TARGET_ROLES)

    assert result.eligible_count == 1


def test_excludes_senior_unrelated_overseas_and_experienced_jobs():
    jobs = pd.DataFrame([
        make_job(
            "1",
            "Senior Director, AI Engineering",
            "Typically 18+ years of engineering experience.",
        ),
        make_job(
            "2",
            "Cloud Data Engineer",
            "Requires 5+ years of data engineering experience.",
            country=None,
        ),
        make_job(
            "3",
            "Software Engineer (Python, AWS)",
            "Requires 2-4 years of software development experience.",
        ),
        make_job(
            "4",
            "Manager Customer Support Engineering",
            "Requires 8-12 years of experience.",
        ),
        make_job(
            "5",
            "Python Backend Engineer",
            "Experience: 2-3 years.",
        ),
        make_job(
            "6",
            "Junior Data Analyst",
            "Requires 2-4 years of analyst experience.",
        ),
    ])

    result = filter_jobs_for_candidate(jobs, TARGET_ROLES)

    assert result.eligible_count == 0
    assert result.excluded_count == 6


def test_excludes_mandatory_advanced_degree_jobs():
    jobs = pd.DataFrame([
        make_job(
            "slb",
            "Data Scientist",
            (
                "A Master of Science degree in Data Science "
                "with a thesis or research projects."
            ),
        ),
        make_job(
            "ion",
            "Data Scientist PhD",
            "PhD is mandatory in Computer Science.",
        ),
    ])

    result = filter_jobs_for_candidate(jobs, TARGET_ROLES)

    assert result.eligible_count == 0
    assert result.excluded_count == 2

    reasons = result.excluded_jobs["exclusion_reasons"].tolist()

    assert any(
        "Master's degree" in reason
        for group in reasons for reason in group
    )
    assert any(
        "PhD" in reason
        for group in reasons for reason in group
    )


def test_rejects_unsupported_target_roles():
    jobs = pd.DataFrame([
        make_job("1", "Junior Data Analyst"),
    ])

    with pytest.raises(JobEligibilityError):
        filter_jobs_for_candidate(jobs, ["Graphic Designer"])


def test_detects_experience_when_requirement_uses_plus_years():
    assert extract_minimum_required_experience(
        "4+ years of experience in Data Science."
    ) == 4.0


def test_detects_innover_seven_year_requirement():
    assert extract_minimum_required_experience(
        "Mandatory Skills & Experience (7+ Years)\n"
        "7+ years of experience in data analysis and statistical analysis."
    ) == 7.0


def test_detects_fedex_standard_experience_requirement():
    description = """
    Relevant work experience as in the following number of years:
    Standard I: Two (2) years
    Standard II: Three (3) years
    Senior I: Four (4) years
    Senior II: Five (5) years
    """

    assert extract_minimum_required_experience(description) == 5.0


def test_does_not_use_incidental_junior_word_to_override_seniority():
    description = (
        "This team works with junior engineers. "
        "The selected candidate should have strong professional skills."
    )

    assert is_senior_job(
        "Data Analyst",
        "Mid-Senior level",
        description,
    )


def test_experience_and_equivalency_detection_for_corning():
    description = (
        "Typically requires 2-4 years of experience in data science, "
        "or equivalent academic and project experience."
    )

    assert extract_minimum_required_experience(description) == 2.0
    assert "equivalent academic and project experience" in description


def test_corning_equivalency_accepts_degree_and_project_evidence():
    from app.services.candidate_profile import CandidateProfile

    profile = CandidateProfile(
        skills=["Python", "SQL", "Machine Learning"],
        education_evidence=["2026 B.Tech AI & ML graduate"],
        sections={
            "projects": (
                "Developed Python-based machine learning projects "
                "involving data cleaning, exploratory data analysis, "
                "feature engineering and model evaluation."
            )
        },
    )

    jobs = pd.DataFrame([
        make_job(
            "corning",
            "AI/ML Data Scientist/Engineer",
            (
                "Typically requires 2-4 years of experience in data "
                "science, analytics, or machine learning roles, "
                "or equivalent academic and project experience."
            ),
        )
    ])

    result = filter_jobs_for_candidate(
        jobs,
        TARGET_ROLES,
        candidate_profile=profile,
    )

    assert result.eligible_count == 1
    assert bool(
        result.eligible_jobs.iloc[0]["experience_equivalency_applied"]
    ) is True


def test_equivalency_is_not_applied_without_project_evidence():
    from app.services.candidate_profile import CandidateProfile

    profile = CandidateProfile(
        education_evidence=["2026 B.Tech AI & ML graduate"],
        sections={},
    )

    jobs = pd.DataFrame([
        make_job(
            "corning",
            "AI/ML Data Scientist/Engineer",
            (
                "Typically requires 2-4 years of experience in "
                "machine learning, or equivalent academic and "
                "project experience."
            ),
        )
    ])

    result = filter_jobs_for_candidate(
        jobs,
        TARGET_ROLES,
        candidate_profile=profile,
    )

    assert result.eligible_count == 0
    assert any(
        "years of experience" in reason
        for reason in result.excluded_jobs.iloc[0]["exclusion_reasons"]
    )


def test_equivalency_is_not_applied_without_bachelors_evidence():
    from app.services.candidate_profile import CandidateProfile

    profile = CandidateProfile(
        education_evidence=[],
        sections={
            "projects": (
                "Developed several Python machine learning projects "
                "with data cleaning, feature engineering and evaluation."
            )
        },
    )

    jobs = pd.DataFrame([
        make_job(
            "corning",
            "AI/ML Data Scientist/Engineer",
            (
                "Typically requires 2-4 years of experience, "
                "or equivalent academic and project experience."
            ),
        )
    ])

    result = filter_jobs_for_candidate(
        jobs,
        TARGET_ROLES,
        candidate_profile=profile,
    )

    assert result.eligible_count == 0


def test_equivalency_does_not_override_a_two_year_requirement_without_alternative():
    from app.services.candidate_profile import CandidateProfile

    profile = CandidateProfile(
        education_evidence=["2026 B.Tech AI & ML graduate"],
        sections={
            "projects": (
                "Developed Python machine learning projects involving "
                "model training, feature engineering and evaluation."
            )
        },
    )

    jobs = pd.DataFrame([
        make_job(
            "ordinary",
            "Data Scientist",
            "Requires 2-4 years of professional data science experience.",
        )
    ])

    result = filter_jobs_for_candidate(
        jobs,
        TARGET_ROLES,
        candidate_profile=profile,
    )

    assert result.eligible_count == 0
