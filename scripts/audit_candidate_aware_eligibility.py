import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.services.candidate_profile import CandidateProfile
from app.services.job_eligibility import filter_jobs_for_candidate


jobs = pd.read_parquet("data/processed/jobs_cleaned.parquet")

jobs = jobs[
    jobs["eligible_for_semantic_matching"].eq(True)
].copy()

test_profile = CandidateProfile(
    skills=[
        "Python",
        "SQL",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Machine Learning",
    ],
    education_evidence=[
        "2026 B.Tech AI & ML graduate"
    ],
    sections={
        "projects": (
            "Developed Python-based data analysis and machine learning "
            "projects involving data cleaning, exploratory data analysis, "
            "feature engineering, model training and evaluation."
        )
    },
    target_roles=[
        "Junior Data Analyst",
        "Junior Data Scientist",
        "Entry-Level Machine Learning Engineer",
    ],
)

result = filter_jobs_for_candidate(
    jobs=jobs,
    target_roles=test_profile.target_roles,
    candidate_profile=test_profile,
    max_required_experience_years=1,
    preferred_country="India",
)

print("\nCANDIDATE-AWARE ELIGIBILITY AUDIT")
print("=" * 75)
print("Original indexed jobs:", result.total_jobs)
print("Eligible jobs:", result.eligible_count)
print("Excluded jobs:", result.excluded_count)

print("\nACCEPTED JOBS")
print("-" * 75)

columns = [
    "id",
    "title",
    "company",
    "location",
    "matched_target_role",
    "required_experience_min_years",
    "experience_equivalency_applied",
]

print(
    result.eligible_jobs[columns]
    .to_string(index=False)
)

print("\nEXPERIENCE EQUIVALENCY DECISIONS")
print("-" * 75)

print(
    result.eligible_jobs[
        result.eligible_jobs["experience_equivalency_applied"]
    ][["id", "title", "company"]].to_string(index=False)
)

print("\nREQUIRED REGRESSION CHECKS")
print("-" * 75)

expected_excluded = {
    "linkedin_4371957148": "UiPath",
    "linkedin_4371039081": "FedEx",
    "linkedin_4392867623": "Innover",
    "linkedin_4404120063": "SLB",
    "linkedin_4404110944": "SLB",
    "linkedin_4404127001": "SLB",
    "linkedin_4404112449": "SLB",
    "linkedin_4371890342": "ION",
}

eligible_ids = set(result.eligible_jobs["id"].astype(str))
excluded_ids = set(result.excluded_jobs["id"].astype(str))

for job_id, company in expected_excluded.items():
    status = "PASS" if job_id in excluded_ids else "FAIL"
    print(f"{status} | {company} | {job_id}")

corning_id = "linkedin_4378807531"
corning = result.eligible_jobs[
    result.eligible_jobs["id"].astype(str).eq(corning_id)
]

corning_passed = (
    not corning.empty
    and bool(corning.iloc[0]["experience_equivalency_applied"])
)

print(
    f"{'PASS' if corning_passed else 'FAIL'} | "
    "Corning academic/project equivalency"
)

omni_id = "linkedin_4328828438"
omni_passed = omni_id in eligible_ids

print(
    f"{'PASS' if omni_passed else 'FAIL'} | "
    "OmniReach fresher opportunity"
)

output_dir = Path("data/processed/eligibility_audit")
output_dir.mkdir(parents=True, exist_ok=True)

result.eligible_jobs.to_csv(
    output_dir / "candidate_aware_eligible_jobs.csv",
    index=False,
)

result.excluded_jobs.to_csv(
    output_dir / "candidate_aware_excluded_jobs.csv",
    index=False,
)

print("\nSaved candidate-aware audit artifacts.")
