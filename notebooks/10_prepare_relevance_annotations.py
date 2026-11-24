from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANKING_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tfidf_ranked_jobs.csv"
)

JOBS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "jobs_cleaned.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "tfidf_top10_for_review.csv"
)


def main():

    rankings = pd.read_csv(RANKING_PATH)
    jobs = pd.read_parquet(JOBS_PATH)

    if rankings["id"].duplicated().any():
        raise ValueError("Duplicate job IDs in ranking results.")

    top_jobs = rankings.head(10).copy()

    top_jobs.insert(
        0,
        "rank",
        range(1, len(top_jobs) + 1)
    )

    annotations = top_jobs.merge(
        jobs[["id", "description"]],
        on="id",
        how="left",
        validate="one_to_one"
    )

    if annotations["description"].isna().any():
        raise ValueError("Some job descriptions could not be retrieved.")

    # Empty fields will be completed during human evaluation.
    annotations["relevance_grade"] = pd.Series(
        pd.NA,
        index=annotations.index,
        dtype="Int64"
    )

    annotations["judgment_reason"] = ""
    annotations["judgment_status"] = "pending"

    columns = [
        "rank",
        "id",
        "title",
        "company",
        "location",
        "seniority_level",
        "employment_type",
        "tfidf_similarity",
        "description",
        "relevance_grade",
        "judgment_reason",
        "judgment_status"
    ]

    annotations[columns].to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEVALUATION DATASET CREATED")
    print("=" * 50)

    print("Records prepared:", len(annotations))
    print("Relevance labels assigned:", annotations["relevance_grade"].notna().sum())
    print("Pending judgments:", annotations["judgment_status"].eq("pending").sum())

    print("\nOutput file:")
    print(OUTPUT_PATH)

    print("\nThe CSV is ready for human annotation.")
    print("No relevance labels have been invented.")


if __name__ == "__main__":
    main()
