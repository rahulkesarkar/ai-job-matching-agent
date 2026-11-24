from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RANKING_PATH = ROOT / "data" / "processed" / "tfidf_ranked_jobs.csv"
JOBS_PATH = ROOT / "data" / "processed" / "jobs_cleaned.parquet"


def main():
    rankings = pd.read_csv(RANKING_PATH)
    jobs = pd.read_parquet(JOBS_PATH)

    top_jobs = rankings.head(10)

    inspected = top_jobs.merge(
        jobs[["id", "description"]],
        on="id",
        how="left",
        validate="one_to_one"
    )

    for rank, (_, job) in enumerate(inspected.iterrows(), start=1):
        print("\n" + "=" * 70)
        print(f"RANK: {rank}")
        print("Job ID:", job["id"])
        print("Title:", job["title"])
        print("Company:", job["company"])
        print("Location:", job["location"])
        print("Seniority:", job["seniority_level"])
        print("TF-IDF score:", round(job["tfidf_similarity"], 4))
        print("\nJOB DESCRIPTION PREVIEW:")
        print(str(job["description"])[:600])

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
