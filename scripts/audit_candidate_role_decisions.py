import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
import pandas as pd

from app.services.job_eligibility import filter_jobs_for_candidate

jobs = pd.read_parquet("data/processed/jobs_cleaned.parquet")
jobs = jobs[jobs["eligible_for_semantic_matching"].eq(True)].copy()

targets = [
    "Junior Data Analyst",
    "Junior Data Scientist",
    "Entry-Level Machine Learning Engineer",
]

audit = filter_jobs_for_candidate(jobs, targets)

eligible = audit.eligible_jobs.set_index("id")
excluded = audit.excluded_jobs.set_index("id")

print("\nELIGIBILITY DECISION AUDIT")
print("=" * 85)

# Review all accepted roles plus known benchmark candidates.
known_companies = [
    "OmniReach",
    "Corning",
    "InfoCepts",
    "UST",
    "UiPath",
    "Apple",
    "Innover",
    "FedEx",
    "PwC",
    "Crownstack",
]

selected = jobs[
    jobs["company"].fillna("").apply(
        lambda value: any(
            name.lower() in value.lower()
            for name in known_companies
        )
    )
]

selected_ids = set(eligible.index.astype(str))
selected_ids.update(excluded.index.astype(str))

selected = selected[
    selected["id"].astype(str).isin(selected_ids)
]

for _, job in selected.iterrows():
    job_id = str(job["id"])
    description = str(job["description"])

    if job_id in eligible.index:
        decision = eligible.loc[job_id]
        status = "ACCEPTED"
    else:
        decision = excluded.loc[job_id]
        status = "EXCLUDED"

    print("\n" + "-" * 85)
    print(status, "|", job["title"], "|", job["company"])
    print("ID:", job_id)
    print("Seniority:", job["seniority_level"])
    print("Role family:", job["role_family_hint"])

    if status == "ACCEPTED":
        print("Matched role:", decision["matched_target_role"])
        print(
            "Detected experience:",
            decision["required_experience_min_years"],
        )
        print(
            "Detected advanced degree:",
            decision["required_advanced_degree"],
        )
    else:
        print("Exclusion reasons:", decision["exclusion_reasons"])

    print("\nEVIDENCE FROM DESCRIPTION:")

    for line in description.splitlines():
        if re.search(
            r"\b(experience|years?|freshers?|graduate|entry.level|"
            r"intern|master|phd|doctorate|research|required|"
            r"qualification|bachelor)\b",
            line,
            re.IGNORECASE,
        ):
            print(" ", line.strip()[:350])

print("\n\nFINAL ACCEPTED COUNT:", audit.eligible_count)
print("FINAL EXCLUDED COUNT:", audit.excluded_count)
