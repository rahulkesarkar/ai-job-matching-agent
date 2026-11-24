import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "role_radar_scraped_jobs.json"


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    df = pd.json_normalize(records)

    print("\n" + "=" * 55)
    print("DATASET GEOGRAPHIC AND ROLE COVERAGE")
    print("=" * 55)

    # 1. Geographic coverage
    print("\n1. GEOGRAPHIC COVERAGE")

    locations = df["location"].fillna("").astype("string").str.strip()

    india_mask = locations.str.contains(
        r"\bIndia\b", case=False, na=False
    )

    missing_mask = locations.eq("")

    print("Total job records:", len(df))
    print("Locations explicitly mentioning India:", int(india_mask.sum()))
    print(
        "Locations not explicitly mentioning India:",
        int((~india_mask & ~missing_mask).sum())
    )
    print("Missing locations:", int(missing_mask.sum()))

    print("\nTop 15 non-India-marked locations:")

    print(
        locations[~india_mask & ~missing_mask]
        .value_counts()
        .head(15)
        .to_string()
    )

    # 2. Job role coverage
    print("\n2. JOB ROLE DISTRIBUTION")

    print(
        df["role_family_hint"]
        .value_counts(dropna=False)
        .to_string()
    )

    # 3. Seniority coverage
    print("\n3. SENIORITY DISTRIBUTION")

    seniority_counts = df["seniority_level"].value_counts(
        dropna=False
    )

    print(seniority_counts.to_string())

    print("\nEntry-level jobs:", int(
        df["seniority_level"].eq("Entry level").sum()
    ))

    print("Internship jobs:", int(
        df["seniority_level"].eq("Internship").sum()
    ))

    # 4. Employment types
    print("\n4. EMPLOYMENT TYPE DISTRIBUTION")

    print(
        df["employment_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    # 5. Remote work information
    print("\n5. REMOTE WORK INDICATOR")

    print(
        df["remote_hint"]
        .value_counts(dropna=False)
        .to_string()
    )

    # 6. Repeated description investigation
    print("\n6. REPEATED DESCRIPTION INVESTIGATION")

    descriptions = (
        df["description"]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    valid = descriptions.ne("")

    description_groups = (
        df.loc[valid]
        .assign(_description=descriptions[valid])
        .groupby("_description")
        .agg(
            records=("id", "size"),
            unique_ids=("id", "nunique"),
            unique_titles=("title", "nunique"),
            unique_companies=("company", "nunique")
        )
    )

    repeated = description_groups[
        description_groups["records"] > 1
    ]

    print("Repeated description groups:", len(repeated))
    print("Records in those groups:", int(repeated["records"].sum()))

    print(
        "Groups used by multiple companies:",
        int((repeated["unique_companies"] > 1).sum())
    )

    print(
        "Groups used across multiple job titles:",
        int((repeated["unique_titles"] > 1).sum())
    )

    # 7. Dataset dates
    print("\n7. DATA COLLECTION DATES")

    dates = pd.to_datetime(df["scraped_at"], errors="coerce")

    print("Earliest scrape date:", dates.min())
    print("Latest scrape date:", dates.max())

    print("\n" + "=" * 55)
    print("COVERAGE AUDIT COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
