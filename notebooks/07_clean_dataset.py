import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs_with_normalized_locations.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs_cleaned.parquet"
)


def main():

    print("\nAI JOB MATCHING AGENT - DATA CLEANING")
    print("=" * 55)

    # 1. Load the processed dataset
    jobs = pd.read_parquet(INPUT_PATH)

    original_count = len(jobs)

    print("\nOriginal records:", original_count)

    # 2. Validate job IDs
    if jobs["id"].isna().any():
        raise ValueError("Missing job IDs found. Investigate before cleaning.")

    # 3. Check whether duplicate IDs contain conflicting records
    duplicate_mask = jobs.duplicated(subset=["id"], keep=False)

    duplicate_groups = jobs.loc[duplicate_mask].groupby("id")

    conflicting_ids = []

    for job_id, group in duplicate_groups:
        if len(group.drop_duplicates()) > 1:
            conflicting_ids.append(job_id)

    if conflicting_ids:
        raise ValueError(
            f"Conflicting duplicate job IDs found: {conflicting_ids[:10]}"
        )

    # 4. Remove verified duplicate job IDs
    duplicate_count = int(jobs["id"].duplicated().sum())

    jobs = jobs.drop_duplicates(
        subset=["id"],
        keep="first"
    ).copy()

    print("Duplicate records removed:", duplicate_count)

    # 5. Identify completely empty columns
    empty_columns = [
        column
        for column in jobs.columns
        if jobs[column].isna().all()
    ]

    print("Completely empty columns removed:", empty_columns)

    jobs = jobs.drop(columns=empty_columns)

    # 6. Measure job-description quality
    description_text = (
        jobs["description"]
        .astype("string")
        .str.strip()
    )

    description_length = description_text.str.len()

    empty_description = (
        description_text.isna()
        | description_text.eq("")
    )

    placeholder_description = description_text.str.fullmatch(
        r"[-–—]+|n/?a|none|null",
        case=False,
        na=False
    )

    short_description = description_length.lt(100).fillna(False)

    quality_status = pd.Series(
        "usable",
        index=jobs.index,
        dtype="string"
    )

    quality_status.loc[short_description] = "short_review"
    quality_status.loc[placeholder_description] = "placeholder"
    quality_status.loc[empty_description] = "empty"

    jobs["description_length_chars"] = description_length
    jobs["description_quality_status"] = quality_status

    jobs["eligible_for_semantic_matching"] = (
        quality_status.eq("usable")
    )

    # 7. Preserve missing categorical information
    # Do not invent missing job functions or other metadata.

    # 8. Save the cleaned dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    jobs.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow"
    )

    # 9. Verify the saved dataset
    verified = pd.read_parquet(
        OUTPUT_PATH,
        columns=["id", "description"]
    )

    if len(verified) != len(jobs):
        raise ValueError("Saved dataset row count does not match.")

    if not verified["id"].is_unique:
        raise ValueError("Duplicate IDs remain in cleaned dataset.")

    # 10. Print the cleaning report
    print("\nCLEANING REPORT")
    print("-" * 55)

    print("Original records:", original_count)
    print("Cleaned records:", len(jobs))
    print("Records removed:", original_count - len(jobs))
    print("Unique job IDs:", jobs["id"].nunique())

    print("\nDescription quality:")

    print(
        jobs["description_quality_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nEligible for semantic matching:",
        int(jobs["eligible_for_semantic_matching"].sum())
    )

    print(
        "Missing job functions:",
        int(jobs["job_function"].isna().sum())
        if "job_function" in jobs.columns
        else "Column unavailable"
    )

    print("\nCleaned dataset saved to:")
    print(OUTPUT_PATH)

    print("\nVALIDATION PASSED")
    print("All retained job IDs are unique.")
    print("Saved record count verified.")

    print("=" * 55)


if __name__ == "__main__":
    main()
