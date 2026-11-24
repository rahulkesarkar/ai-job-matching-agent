import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "role_radar_scraped_jobs.json"

ALIAS_PATH = PROJECT_ROOT / "data" / "reference" / "location_aliases.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "jobs_with_normalized_locations.parquet"


def parse_explicit_india_location(location):
    """Extract city, state and country from an explicit India location."""

    parts = [part.strip() for part in location.split(",")]

    city = parts[0]
    state = parts[-2] if len(parts) >= 3 else pd.NA

    return city, state, "India"


def main():

    # 1. Load original job records
    with DATA_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    jobs = pd.json_normalize(records)

    # 2. Load our manually reviewed location aliases
    aliases = pd.read_csv(ALIAS_PATH, dtype="string")

    # Ensure that every original alias is unique
    if aliases["original_location"].duplicated().any():
        raise ValueError("Duplicate location aliases found.")

    # 3. Standardize whitespace without modifying the source file
    jobs["location"] = (
        jobs["location"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # 4. Match job locations against our alias table
    jobs = jobs.merge(
        aliases,
        how="left",
        left_on="location",
        right_on="original_location",
        validate="many_to_one",
        indicator="_alias_match"
    )

    # 5. Set the initial classification
    jobs["normalization_method"] = "unresolved"

    missing_mask = jobs["location"].isna()
    alias_mask = jobs["_alias_match"].eq("both")

    jobs.loc[missing_mask, "normalization_method"] = "missing"
    jobs.loc[alias_mask, "normalization_method"] = "alias_mapping"

    # 6. Parse locations explicitly ending in India
    india_mask = (
        jobs["location"]
        .str.contains(r",\s*India$", case=False, na=False)
        & ~alias_mask
    )

    for index in jobs.index[india_mask]:
        city, state, country = parse_explicit_india_location(
            jobs.at[index, "location"]
        )

        jobs.at[index, "normalized_city"] = city
        jobs.at[index, "normalized_state"] = state
        jobs.at[index, "normalized_country"] = country
        jobs.at[index, "location_type"] = "city_or_locality"

        jobs.at[index, "normalization_method"] = "explicit_country"

    # 7. Count normalization results
    print("\nLOCATION NORMALIZATION RESULTS")
    print("=" * 55)

    print("Total records:", len(jobs))

    print(
        "Alias-mapped records:",
        int(alias_mask.sum())
    )

    print(
        "Explicitly identified Indian locations:",
        int(jobs["normalized_country"].eq("India").sum())
    )

    print(
        "Unresolved locations:",
        int(jobs["normalization_method"].eq("unresolved").sum())
    )

    print(
        "Missing locations:",
        int(missing_mask.sum())
    )

    # 8. Review unresolved locations
    unresolved = jobs[
        jobs["normalization_method"].eq("unresolved")
    ]

    print("\nTOP UNRESOLVED LOCATIONS")

    print(
        unresolved["location"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    # 9. Save the processed dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    jobs = jobs.drop(
        columns=["original_location", "_alias_match"]
    )

    jobs.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow"
    )

    print("\nProcessed dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
