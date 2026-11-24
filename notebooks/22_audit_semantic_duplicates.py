import re
import html
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RANKING_PATH = ROOT / "data/processed/semantic_ranked_jobs.csv"
JOBS_PATH = ROOT / "data/processed/jobs_cleaned.parquet"


def normalize_description(text):
    text = html.unescape(str(text))
    text = text.replace("<![CDATA[", "").replace("]]>", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def main():

    ranking = pd.read_csv(RANKING_PATH)
    jobs = pd.read_parquet(JOBS_PATH)

    top10 = ranking.head(10).merge(
        jobs[["id", "description", "url"]],
        left_on="job_id",
        right_on="id",
        how="left",
        validate="one_to_one"
    )

    top10["normalized_description"] = (
        top10["description"].map(normalize_description)
    )

    top10["fingerprint"] = (
        top10["normalized_description"].map(
            lambda text: hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()
        )
    )

    print("\nSEMANTIC DUPLICATE AUDIT")
    print("=" * 60)

    groups = top10.groupby(
        ["company", "title", "fingerprint"]
    )

    duplicate_groups = []

    for (company, title, fingerprint), group in groups:

        if len(group) > 1:

            duplicate_groups.append(group)

            print("\nCompany:", company)
            print("Title:", title)
            print("Identical description postings:", len(group))

            for _, row in group.iterrows():
                print("Job ID:", row["job_id"])
                print("Location:", row["location"])
                print("URL:", row["url"])

    print("\nSUMMARY")
    print("-" * 60)

    print("Top 10 postings:", len(top10))
    print("Exact duplicate description groups:", len(duplicate_groups))
    print(
        "Postings belonging to duplicate groups:",
        sum(len(group) for group in duplicate_groups)
    )

    print("\nAUDIT COMPLETED")


if __name__ == "__main__":
    main()
