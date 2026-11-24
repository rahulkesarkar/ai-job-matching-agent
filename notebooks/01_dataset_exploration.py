import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "role_radar_scraped_jobs.json"


def main():
    print("=" * 60)
    print("AI JOB MATCHING AGENT - DATASET EXPLORATION")
    print("=" * 60)

    # Load the original JSON dataset
    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    # Convert JSON records into a Pandas DataFrame
    df = pd.json_normalize(records)

    # Basic dataset information
    print("\n1. DATASET OVERVIEW")
    print("-" * 40)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    # Inspect data types
    print("\n2. COLUMN DATA TYPES")
    print("-" * 40)

    print(df.dtypes.to_string())

    # Inspect missing and blank values
    print("\n3. MISSING VALUE ANALYSIS")
    print("-" * 40)

    null_counts = df.isna().sum()

    blank_counts = df.select_dtypes(include="object").apply(
        lambda column: column.map(
            lambda value: isinstance(value, str) and not value.strip()
        ).sum()
    )

    audit = pd.DataFrame({
        "null_count": null_counts,
        "blank_count": blank_counts.reindex(
            df.columns, fill_value=0
        )
    })

    audit["missing_percentage"] = (
        (audit["null_count"] + audit["blank_count"])
        / len(df) * 100
    ).round(2)

    print(audit.to_string())

    # Check duplicate records
    print("\n4. DUPLICATE ANALYSIS")
    print("-" * 40)

    print("Exact duplicate rows:", df.duplicated().sum())

    if "id" in df.columns:
        print("Duplicate job IDs:", df["id"].duplicated().sum())

    # Analyze job description lengths
    print("\n5. JOB DESCRIPTION ANALYSIS")
    print("-" * 40)

    if "description" in df.columns:
        description_lengths = (
            df["description"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.len()
        )

        print(description_lengths.describe())

        print(
            "\nDescriptions shorter than 100 characters:",
            (description_lengths < 100).sum()
        )

    # Display a sample without printing the entire description
    print("\n6. SAMPLE JOB RECORD")
    print("-" * 40)

    if len(df) > 0:
        sample = df.iloc[0]

        for column in df.columns:
            value = sample[column]

            if column == "description":
                value = str(value)[:300] + "..."

            print(f"{column}: {value}")

    print("\n" + "=" * 60)
    print("DATASET EXPLORATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
