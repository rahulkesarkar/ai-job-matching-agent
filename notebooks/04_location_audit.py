import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "role_radar_scraped_jobs.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "location_value_counts.csv"


def main():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    df = pd.json_normalize(records)

    locations = (
        df["location"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    location_counts = (
        locations
        .value_counts(dropna=False)
        .rename_axis("original_location")
        .reset_index(name="job_count")
    )

    location_counts["contains_india"] = (
        location_counts["original_location"]
        .astype("string")
        .str.contains(r"\bIndia\b", case=False, na=False)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    location_counts.to_csv(OUTPUT_PATH, index=False)

    locations_for_review = location_counts[
        ~location_counts["contains_india"]
    ]

    print("\nLOCATION AUDIT")
    print("=" * 60)

    print("Total job records:", len(df))
    print("Unique location values:", len(location_counts))

    print(
        "Records containing India:",
        int(location_counts.loc[
            location_counts["contains_india"], "job_count"
        ].sum())
    )

    print(
        "Records requiring location review:",
        int(locations_for_review["job_count"].sum())
    )

    print(
        "\nLocation values requiring review:",
        len(locations_for_review)
    )

    print("\nLOCATION VALUES AND COUNTS")
    print("-" * 60)

    print(
        locations_for_review.to_string(index=False)
    )

    print("\nSaved location audit to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
