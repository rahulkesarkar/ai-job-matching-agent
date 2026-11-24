from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.candidate_profile import CandidateProfile
from app.services.job_matching_engine import (
    JobMatchingEngine,
    JobMatchingError,
)


def create_engine(jobs):
    """Create an engine without loading the actual ML models."""
    engine = JobMatchingEngine.__new__(JobMatchingEngine)

    engine.eligible_jobs = pd.DataFrame(jobs)
    engine.index = SimpleNamespace(ntotal=10095)
    engine.model = SimpleNamespace(device="cpu")

    return engine


def test_both_retrievers_only_receive_eligible_job_ids(monkeypatch):
    jobs = [
        {
            "id": "good",
            "title": "Junior Data Analyst",
            "company": "Good Company",
            "location": "Mumbai, Maharashtra, India",
            "description": "Python and SQL data analysis.",
            "seniority_level": "Entry level",
            "normalized_country": "India",
        },
        {
            "id": "senior",
            "title": "Senior Data Scientist",
            "company": "Senior Company",
            "location": "Mumbai, Maharashtra, India",
            "description": "18+ years of data science experience.",
            "seniority_level": "Director",
            "normalized_country": "India",
        },
        {
            "id": "experienced",
            "title": "Data Analyst",
            "company": "Experienced Company",
            "location": "Mumbai, Maharashtra, India",
            "description": "Requires 3+ years of experience.",
            "seniority_level": "Associate",
            "normalized_country": "India",
        },
    ]

    engine = create_engine(jobs)

    profile = CandidateProfile(
        skills=["Python", "SQL"],
        target_roles=["Junior Data Analyst"],
        matching_text="Junior Data Analyst Python SQL data analysis",
    )

    received_ids = {}

    def fake_tfidf(candidate_text, allowed_job_ids):
        received_ids["tfidf"] = allowed_job_ids

        return pd.DataFrame([
            {
                "id": "good",
                "title": "Junior Data Analyst",
                "company": "Good Company",
                "location": "Mumbai, Maharashtra, India",
                "tfidf_similarity": 0.9,
            }
        ])

    def fake_semantic(candidate_text, allowed_job_ids):
        received_ids["semantic"] = allowed_job_ids

        return pd.DataFrame([
            {
                "job_id": "good",
                "title": "Junior Data Analyst",
                "company": "Good Company",
                "location": "Mumbai, Maharashtra, India",
                "description": "Python and SQL data analysis.",
                "semantic_similarity": 0.9,
                "semantic_rank": 1,
            }
        ]), 1

    monkeypatch.setattr(engine, "rank_tfidf", fake_tfidf)
    monkeypatch.setattr(engine, "rank_semantic", fake_semantic)

    result = engine.match(profile)

    assert received_ids["tfidf"] == {"good"}
    assert received_ids["semantic"] == {"good"}

    assert result.eligible_job_count == 1
    assert result.excluded_job_count == 2

    assert result.tfidf_ranked["id"].tolist() == ["good"]
    assert result.semantic_ranked["job_id"].tolist() == ["good"]


def test_matching_rejects_when_no_jobs_are_eligible(monkeypatch):
    jobs = [
        {
            "id": "senior",
            "title": "Senior Data Scientist",
            "company": "Senior Company",
            "location": "Mumbai, Maharashtra, India",
            "description": "18+ years of experience.",
            "seniority_level": "Director",
            "normalized_country": "India",
        }
    ]

    engine = create_engine(jobs)

    profile = CandidateProfile(
        skills=["Python"],
        target_roles=["Junior Data Analyst"],
        matching_text="Junior Data Analyst Python",
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("Ranking must not run without eligible jobs.")

    monkeypatch.setattr(engine, "rank_tfidf", should_not_run)
    monkeypatch.setattr(engine, "rank_semantic", should_not_run)

    with pytest.raises(JobMatchingError, match="No jobs passed"):
        engine.match(profile)
