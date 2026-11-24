from pathlib import Path

import pandas as pd

from app.services.recommendation_diversifier import (
    diversify_recommendations
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

RANKING_PATH = DATA_DIR / "semantic_ranked_jobs.csv"

JOBS_PATH = DATA_DIR / "jobs_cleaned.parquet"

OUTPUT_PATH = DATA_DIR / "semantic_ranked_jobs_diversified.csv"


def main():

    print("\nRECOMMENDATION DIVERSIFICATION")
    print("=" * 60)

    # 1. Load original semantic rankings
    rankings = pd.read_csv(RANKING_PATH)

    jobs = pd.read_parquet(JOBS_PATH)

    if rankings["job_id"].duplicated().any():
        raise ValueError("Duplicate job IDs in original rankings.")

    # 2. Attach the original job descriptions
    ranking_with_descriptions = rankings.merge(
        jobs[["id", "description"]],
        left_on="job_id",
        right_on="id",
        how="left",
        validate="one_to_one"
    )

    if ranking_with_descriptions["description"].isna().any():
        raise ValueError("Some job descriptions could not be matched.")

    print("Original ranked jobs:", len(rankings))

    # 3. Apply exact duplicate diversification
    diversified = diversify_recommendations(
        ranking_with_descriptions
    )

    # 4. Validate retained representatives
    if diversified["job_id"].duplicated().any():
        raise ValueError("Duplicate representatives detected.")

    represented_ids = list(diversified["job_id"].astype(str))

    for alternate_ids in diversified["alternate_job_ids"]:
        if isinstance(alternate_ids, str) and alternate_ids.strip():
            represented_ids.extend(
                alternate_ids.split(", ")
            )

    original_ids = set(rankings["job_id"].astype(str))

    if set(represented_ids) != original_ids:
        raise ValueError(
            "Some original job IDs are missing or incorrectly mapped."
        )

    if len(represented_ids) != len(original_ids):
        raise ValueError(
            "Some original job IDs have been represented more than once."
        )

    # Validate that every original posting belongs to one group.
    if diversified["duplicate_posting_count"].sum() != len(rankings):
        raise ValueError("Duplicate group counts do not match input records.")

    # 5. Save the diversified ranking
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    diversified.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 6. Display the new top 10
    print("\nDIVERSIFICATION RESULTS")
    print("-" * 60)

    print("Original postings:", len(rankings))
    print("Recommendation representatives:", len(diversified))
    print(
        "Postings grouped as alternatives:",
        len(rankings) - len(diversified)
    )

    print("\nDIVERSIFIED TOP 10")

    display_columns = [
        "diversified_rank",
        "semantic_rank",
        "job_id",
        "title",
        "company",
        "location",
        "semantic_similarity",
        "duplicate_posting_count",
        "alternate_job_ids"
    ]

    print(
        diversified[display_columns]
        .head(10)
        .round({"semantic_similarity": 4})
        .to_string(index=False)
    )

    print("\nOriginal job IDs preserved:", len(represented_ids))

    print("\nSaved diversified ranking:")
    print(OUTPUT_PATH)

    print("\nALL DIVERSIFICATION VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
