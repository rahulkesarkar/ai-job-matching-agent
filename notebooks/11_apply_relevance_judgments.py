from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "processed" / "tfidf_top10_for_review.csv"

judgments = {
    1: (
        2,
        "Entry-level Data Scientist role. Strong overlap with Python, SQL, EDA and machine learning."
    ),
    2: (
        2,
        "Description explicitly welcomes freshers and early-career engineers. Strong Python, data engineering and ML alignment."
    ),
    3: (
        0,
        "PhD is explicitly mandatory. Candidate has a B.Tech degree."
    ),
    4: (
        0,
        "Highly experienced senior data science position with leadership responsibilities."
    ),
    5: (
        2,
        "Junior AI/ML role. Bachelor's degree accepted and equivalent academic and project experience considered."
    ),
    6: (
        1,
        "Relevant Python and machine learning responsibilities, but the description asks for an experienced engineer."
    ),
    7: (
        0,
        "Director-level role requiring advanced production ML and leadership experience."
    ),
    8: (
        0,
        "Same job description and requirements as rank 7, with a different location."
    ),
    9: (
        0,
        "Requires 4-7 years of hands-on AI/ML experience."
    ),
    10: (
        0,
        "Explicitly requires 5-6 years of professional experience."
    )
}


def main():

    df = pd.read_csv(INPUT_PATH)

    expected_ranks = set(judgments.keys())
    actual_ranks = set(df["rank"])

    if actual_ranks != expected_ranks:
        raise ValueError("Unexpected ranks in annotation dataset.")

    if len(df) != 10:
        raise ValueError("Expected exactly 10 evaluation records.")

    df["relevance_grade"] = df["rank"].map(
        lambda rank: judgments[rank][0]
    )

    df["judgment_reason"] = df["rank"].map(
        lambda rank: judgments[rank][1]
    )

    df["judgment_status"] = "reviewed"

    df.to_csv(INPUT_PATH, index=False)

    print("\nRELEVANCE ANNOTATIONS SAVED")
    print("=" * 55)

    print(
        df[
            ["rank", "company", "relevance_grade", "judgment_status"]
        ].to_string(index=False)
    )

    print("\nGrade distribution:")
    print(
        df["relevance_grade"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nReviewed records:", len(df))
    print("Pending records:", (df["judgment_status"] == "pending").sum())

    print("\nSaved to:", INPUT_PATH)


if __name__ == "__main__":
    main()
