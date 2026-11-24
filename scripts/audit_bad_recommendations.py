import pandas as pd

jobs = pd.read_parquet("data/processed/jobs_cleaned.parquet")

bad_result_ids = [
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

selected = jobs[
    jobs["id"].astype(str).isin(bad_result_ids)
].copy()

print("\nINCORRECT RECOMMENDATION AUDIT")
print("=" * 80)

for _, job in selected.iterrows():
    print("\n" + "-" * 80)
    print("JOB:", job["title"])
    print("COMPANY:", job["company"])
    print("LOCATION:", job["location"])
    print("NORMALIZED COUNTRY:", job["normalized_country"])
    print("SENIORITY:", job["seniority_level"])
    print("ROLE FAMILY:", job["role_family_hint"])
    print("EMPLOYMENT:", job["employment_type"])
    print("\nDESCRIPTION:")
    print(str(job["description"])[:1800])

print("\n\nDATASET SENIORITY DISTRIBUTION")
print("=" * 80)
print(jobs["seniority_level"].fillna("Missing").value_counts().to_string())

print("\n\nDATASET ROLE FAMILY DISTRIBUTION")
print("=" * 80)
print(jobs["role_family_hint"].fillna("Missing").value_counts().to_string())

print("\n\nDATASET COUNTRY DISTRIBUTION")
print("=" * 80)
print(jobs["normalized_country"].fillna("Unclassified").value_counts().to_string())

