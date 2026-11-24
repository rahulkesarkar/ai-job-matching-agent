import hashlib
from pathlib import Path

import pandas as pd

from app.services.recommendation_diversifier import (
    normalize_description,
    normalize_label
)


ROOT = Path(__file__).resolve().parents[1]

RANKING_PATH = ROOT / "data/processed/semantic_ranked_jobs.csv"
JOBS_PATH = ROOT / "data/processed/jobs_cleaned.parquet"
OUTPUT_PATH = ROOT / "data/processed/diversification_location_audit.csv"


def main():

    print("\nDIVERSIFICATION LOCATION AUDIT")
    print("=" * 60)

    rankings = pd.read_csv(RANKING_PATH)
    jobs = pd.read_parquet(JOBS_PATH)

    # Remove overlapping columns from rankings.
    # Use normalized location fields from the cleaned dataset.
    rankings = rankings.drop(
        columns=["normalized_city", "normalized_country"],
        errors="ignore"
    )

    records = rankings.merge(
        jobs[
            [
                "id",
                "description",
                "normalized_city",
                "normalized_state",
                "normalized_country"
            ]
        ],
        left_on="job_id",
        right_on="id",
        how="left",
        validate="one_to_one"
    )

    if records["description"].isna().any():
        raise ValueError("Missing job descriptions.")

    # Normalize grouping fields
    records["_company"] = records["company"].map(normalize_label)
    records["_title"] = records["title"].map(normalize_label)

    descriptions = records["description"].map(
        normalize_description
    )

    records["_description_hash"] = descriptions.map(
        lambda value: hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
    )

    # Build normalized location keys
    def location_key(row):

        city = row["normalized_city"]
        state = row["normalized_state"]
        country = row["normalized_country"]

        if pd.notna(city) and pd.notna(country):
            return "|".join(
                str(value).strip().lower()
                for value in [city, state, country]
                if pd.notna(value)
            )

        return normalize_label(row["location"])

    records["_location_key"] = records.apply(
        location_key,
        axis=1
    )

    # Identify exact duplicate groups
    groups = records.groupby(
        ["_company", "_title", "_description_hash"],
        dropna=False
    )

    audit_records = []

    for _, group in groups:

        if len(group) <= 1:
            continue

        locations = sorted(
            group["_location_key"].dropna().unique()
        )

        audit_records.append({
            "company": group.iloc[0]["company"],
            "title": group.iloc[0]["title"],
            "posting_count": len(group),
            "extra_postings": len(group) - 1,
            "unique_locations": len(locations),
            "locations": " | ".join(
                group["location"].fillna("Missing").astype(str).unique()
            ),
            "job_ids": ", ".join(group["job_id"].astype(str))
        })

    audit = pd.DataFrame(audit_records)

    if audit.empty:
        print("No duplicate groups found.")
        return

    cross_location = audit[
        audit["unique_locations"] > 1
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Eligible ranked jobs:", len(records))
    print("Exact duplicate groups:", len(audit))
    print(
        "Additional postings grouped:",
        int(audit["extra_postings"].sum())
    )

    print(
        "Groups spanning multiple locations:",
        len(cross_location)
    )

    print("\nCROSS-LOCATION DUPLICATE GROUPS")

    print(
        cross_location.to_string(index=False)
    )

    print("\nSaved audit:", OUTPUT_PATH)
    print("\nLOCATION AUDIT COMPLETED")


if __name__ == "__main__":
    main()
