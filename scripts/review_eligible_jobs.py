from pathlib import Path
import re
import pandas as pd

root = Path(__file__).resolve().parents[1]

eligible = pd.read_csv(
    root / "data/processed/eligibility_audit/eligible_jobs.csv"
)

jobs = pd.read_parquet(
    root / "data/processed/jobs_cleaned.parquet"
)

pool_path = root / "data/processed/model_comparison_pool.csv"
judgments = pd.read_csv(pool_path) if pool_path.exists() else pd.DataFrame()

print("\nELIGIBLE JOB REVIEW")
print("=" * 85)
print("Eligible jobs:", len(eligible))

for _, job in eligible.iterrows():
    job_id = str(job["id"])
    original = jobs[jobs["id"].astype(str) == job_id]

    if original.empty:
        continue

    original = original.iloc[0]
    description = str(original["description"])

    print("\n" + "-" * 85)
    print("TITLE:", job["title"])
    print("COMPANY:", job["company"])
    print("ID:", job_id)
    print("LOCATION:", job["location"])
    print("ROLE CATEGORY:", job["matched_target_role"])
    print("SENIORITY:", job["seniority_level"])
    print("EXPERIENCE:", job["required_experience_min_years"])

    if not judgments.empty and "id" in judgments.columns:
        reviewed = judgments[
            judgments["id"].astype(str) == job_id
        ]

        if not reviewed.empty:
            label = reviewed.iloc[0]
            print("MANUAL GRADE:", label.get("relevance_grade"))
            print("JUDGMENT:", label.get("judgment_reason"))

    print("DESCRIPTION EVIDENCE:")

    shown = 0
    for line in description.splitlines():
        if re.search(
            r"\b(experience|years?|freshers?|graduate|entry.level|"
            r"intern|skills|required|qualification|bachelor|master|"
            r"phd|research|project|responsibilities)\b",
            line,
            re.IGNORECASE,
        ):
            print(" -", line.strip()[:350])
            shown += 1

        if shown >= 7:
            break

print("\n\nPREVIOUSLY JUDGED RELEVANT JOBS")
print("=" * 85)

if not judgments.empty:
    relevant = judgments[
        pd.to_numeric(judgments["relevance_grade"], errors="coerce") >= 1
    ]

    for _, item in relevant.iterrows():
        print(
            item["id"],
            "|",
            item["title"],
            "| grade:",
            item["relevance_grade"],
            "|",
            item["judgment_reason"],
        )

print("\nREVIEW COMPLETED")
