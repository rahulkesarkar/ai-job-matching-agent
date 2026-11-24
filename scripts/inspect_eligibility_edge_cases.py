import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
import pandas as pd

from app.services.job_eligibility import (
    extract_minimum_required_experience,
    filter_jobs_for_candidate,
)

jobs = pd.read_parquet("data/processed/jobs_cleaned.parquet")
jobs = jobs[jobs["eligible_for_semantic_matching"].eq(True)].copy()

targets = [
    "Junior Data Analyst",
    "Junior Data Scientist",
    "Entry-Level Machine Learning Engineer",
]

result = filter_jobs_for_candidate(jobs, targets)

print("\n1. REVIEW OF ACCEPTED JOBS")
print("=" * 85)

for _, row in result.eligible_jobs.iterrows():
    print(
        f"\n{row['title']} | {row['company']} | "
        f"{row['seniority_level']} | "
        f"experience={row['required_experience_min_years']}"
    )

    description = str(row["description"])

    for line in description.splitlines():
        if re.search(
            r"\b(phd|doctorate|master|postgraduate|research|"
            r"publication|bachelor|b\.?tech|degree)\b",
            line,
            re.IGNORECASE,
        ):
            print("  QUALIFICATION:", line.strip()[:300])

print("\n\n2. REVIEW OF PROMISING COMPANIES")
print("=" * 85)

companies = [
    "OmniReach",
    "InfoCepts",
    "UST",
    "Apple",
    "FedEx",
    "Crownstack",
    "SLB",
    "ION",
]

for company in companies:
    matches = jobs[
        jobs["company"].fillna("").str.contains(
            company, case=False, regex=False
        )
    ]

    if matches.empty:
        continue

    print(f"\n--- {company} ({len(matches)} postings) ---")

    for _, row in matches.head(5).iterrows():
        job_id = str(row["id"])
        rejected = result.excluded_jobs[
            result.excluded_jobs["id"].astype(str).eq(job_id)
        ]

        accepted = result.eligible_jobs[
            result.eligible_jobs["id"].astype(str).eq(job_id)
        ]

        if not accepted.empty:
            status = "ACCEPTED"
            reasons = []
        elif not rejected.empty:
            status = "EXCLUDED"
            reasons = rejected.iloc[0]["exclusion_reasons"]
        else:
            status = "NOT IN SEMANTIC CORPUS"
            reasons = []

        print(
            f"{status} | {row['title']} | "
            f"{row['seniority_level']} | {reasons}"
        )

        description = str(row["description"])
        for line in description.splitlines():
            if re.search(
                r"\b(fresher|freshers|entry.level|internship|"
                r"0\s*[-–]\s*1\s*years?|"
                r"\d+\s*[-–]\s*\d+\s*years?|"
                r"phd|master|research)\b",
                line,
                re.IGNORECASE,
            ):
                print("  EVIDENCE:", line.strip()[:300])

print("\n\n3. SUSPICIOUS EXPERIENCE DETECTIONS")
print("=" * 85)

for _, row in jobs.iterrows():
    description = str(row["description"])
    required = extract_minimum_required_experience(
        f"{row['title']}\n{description}"
    )

    if required is not None and required >= 50:
        print(
            f"\n{row['title']} | {row['company']} | "
            f"detected={required:g} years"
        )

        for match in re.finditer(
            r".{0,100}\b\d+\+?\s*years?\b.{0,120}",
            description,
            re.IGNORECASE | re.DOTALL,
        ):
            print("  CONTEXT:", re.sub(r"\s+", " ", match.group())[:350])

print("\nEDGE-CASE INVESTIGATION COMPLETED")
