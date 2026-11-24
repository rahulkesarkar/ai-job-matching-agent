import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.services.job_eligibility import filter_jobs_for_candidate


ROOT = "data/processed/jobs_cleaned.parquet"

TARGET_ROLES = [
    "Junior Data Analyst",
    "Junior Data Scientist",
    "Entry-Level Machine Learning Engineer",
]


jobs = pd.read_parquet(ROOT)

# Match against the same corpus used by our retrieval engine.
jobs = jobs[
    jobs["eligible_for_semantic_matching"].eq(True)
].copy()

result = filter_jobs_for_candidate(
    jobs=jobs,
    target_roles=TARGET_ROLES,
    max_required_experience_years=1,
    preferred_country="India",
)

print("\nREAL DATASET ELIGIBILITY AUDIT")
print("=" * 75)

print("Original eligible corpus:", result.total_jobs)
print("Jobs passing eligibility filters:", result.eligible_count)
print("Jobs excluded:", result.excluded_count)

print(
    "Eligible percentage:",
    round(result.eligible_count / result.total_jobs * 100, 2),
    "%",
)

print("\nELIGIBLE JOBS BY TARGET CATEGORY")
print("-" * 75)

print(
    result.eligible_jobs["matched_target_role"]
    .value_counts()
    .to_string()
)

print("\nELIGIBLE JOBS BY SENIORITY")
print("-" * 75)

print(
    result.eligible_jobs["seniority_level"]
    .fillna("Missing")
    .value_counts()
    .to_string()
)

print("\nEXPLICIT EXPERIENCE REQUIREMENTS")
print("-" * 75)

experience = result.eligible_jobs[
    "required_experience_min_years"
]

print("Jobs without explicit requirements:", experience.isna().sum())
print("Jobs with requirements:", experience.notna().sum())

print("\nTOP 30 ELIGIBLE JOBS")
print("-" * 75)

columns = [
    "id",
    "title",
    "company",
    "location",
    "seniority_level",
    "matched_target_role",
    "required_experience_min_years",
]

print(
    result.eligible_jobs[columns]
    .head(30)
    .to_string(index=False)
)

print("\nMOST COMMON EXCLUSION REASONS")
print("-" * 75)

reasons = (
    result.excluded_jobs["exclusion_reasons"]
    .explode()
    .value_counts()
)

print(reasons.to_string())

print("\nAUDIT OF PREVIOUSLY INCORRECT RECOMMENDATIONS")
print("-" * 75)

previous_ids = [
    "linkedin_4366544387",
    "linkedin_4402880781",
    "linkedin_4392471636",
    "linkedin_4405651873",
    "linkedin_4388787599",
    "linkedin_4371992536",
    "linkedin_4402976861",
    "linkedin_4382272487",
    "linkedin_4342531958",
    "linkedin_4393562004",
]

excluded = result.excluded_jobs.set_index("id")

for job_id in previous_ids:
    if job_id in excluded.index:
        job = excluded.loc[job_id]
        print(job_id, "| EXCLUDED |", job["exclusion_reasons"])
    elif job_id in set(result.eligible_jobs["id"]):
        print(job_id, "| PASSED FILTERS | Requires further relevance review")
    else:
        print(job_id, "| NOT FOUND IN SEMANTIC CORPUS")

# Save reproducible audit outputs.
output_dir = "data/processed/eligibility_audit"

import os
os.makedirs(output_dir, exist_ok=True)

result.eligible_jobs.to_csv(
    f"{output_dir}/eligible_jobs.csv",
    index=False,
)

result.excluded_jobs.to_csv(
    f"{output_dir}/excluded_jobs.csv",
    index=False,
)

print("\nSaved audit files to:", output_dir)
print("\nELIGIBILITY AUDIT COMPLETED")
