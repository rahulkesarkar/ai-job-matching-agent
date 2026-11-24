import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "role_radar_scraped_jobs.json"


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    df = pd.json_normalize(records)

    print("\n1. COMPLETELY EMPTY COLUMNS")
    empty_columns = df.columns[df.isna().all()].tolist()
    print(empty_columns)

    print("\n2. DUPLICATE JOB ID INVESTIGATION")

    id_counts = df["id"].value_counts()
    duplicate_ids = id_counts[id_counts > 1]

    duplicate_rows = df[
        df["id"].duplicated(keep=False)
    ]

    print("Number of duplicated ID groups:", len(duplicate_ids))
    print("Number of extra rows:", int(df["id"].duplicated().sum()))
    print("Exact duplicate rows:", int(df.duplicated().sum()))

    identical_groups = 0

    for job_id, group in duplicate_rows.groupby("id"):
        unique_records = group.drop(columns=["id"]).drop_duplicates()

        if len(unique_records) == 1:
            identical_groups += 1

    print(
        "Duplicate ID groups with identical records:",
        identical_groups
    )

    print("\nDuplicate job examples:")

    print(
        duplicate_rows[
            ["id", "title", "company", "location"]
        ]
        .drop_duplicates()
        .head(10)
        .to_string(index=False)
    )

    print("\n3. SHORT JOB DESCRIPTIONS")

    description_lengths = (
        df["description"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.len()
    )

    short_mask = description_lengths < 100
    short_jobs = df.loc[short_mask]

    print("Number of short descriptions:", len(short_jobs))

    for index, row in short_jobs.iterrows():
        print("\nJob ID:", row["id"])
        print("Title:", row["title"])
        print("Company:", row["company"])
        print("Description length:", description_lengths.loc[index])
        print("Description:", repr(row["description"]))

    print("\n4. REPEATED JOB DESCRIPTIONS")

    description_text = (
        df["description"]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    description_counts = description_text.value_counts()
    repeated_descriptions = description_counts[
        description_counts > 1
    ]

    print(
        "Distinct descriptions used more than once:",
        len(repeated_descriptions)
    )

    print(
        "Total records belonging to repeated descriptions:",
        int(repeated_descriptions.sum())
    )

    print("\n5. TOP JOB LOCATIONS")

    print(
        df["location"]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

    print("\n6. JOB SENIORITY DISTRIBUTION")

    print(
        df["seniority_level"]
        .value_counts(dropna=False)
        .head(15)
        .to_string()
    )

    print("\n7. SCRAPING DATE DISTRIBUTION")

    print(
        df["scraped_at"]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()
