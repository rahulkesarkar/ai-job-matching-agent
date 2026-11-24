import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data" / "processed" / "tfidf_top10_for_review.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "tfidf_pilot_metrics.json"
)


def precision_at_k(relevance, k):
    relevant = sum(grade >= 1 for grade in relevance[:k])
    return relevant / k


def recall_at_k(relevance, k):
    total_relevant = sum(grade >= 1 for grade in relevance)

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        grade >= 1 for grade in relevance[:k]
    )

    return retrieved_relevant / total_relevant


def mean_reciprocal_rank(relevance):
    for rank, grade in enumerate(relevance, start=1):
        if grade >= 1:
            return 1 / rank

    return 0.0


def dcg_at_k(relevance, k):
    score = 0.0

    for rank, grade in enumerate(relevance[:k], start=1):
        gain = (2 ** grade) - 1
        discount = math.log2(rank + 1)
        score += gain / discount

    return score


def ndcg_at_k(relevance, k):
    actual_dcg = dcg_at_k(relevance, k)

    ideal_relevance = sorted(
        relevance,
        reverse=True
    )

    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def main():

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise ValueError("Evaluation dataset is empty.")

    if df["relevance_grade"].isna().any():
        raise ValueError("Some relevance judgments are missing.")

    if not df["judgment_status"].eq("reviewed").all():
        raise ValueError("Some records have not been reviewed.")

    if not df["rank"].is_monotonic_increasing:
        raise ValueError("Records must be sorted by original rank.")

    relevance = df["relevance_grade"].astype(int).tolist()

    if not set(relevance).issubset({0, 1, 2}):
        raise ValueError("Unexpected relevance grades found.")

    metrics = {
        "evaluation_type": "Top-10 TF-IDF pilot evaluation",
        "reviewed_jobs": len(df),
        "highly_relevant_jobs": relevance.count(2),
        "partially_relevant_jobs": relevance.count(1),
        "not_relevant_jobs": relevance.count(0),
        "binary_relevance_threshold": "grade >= 1",
        "precision_at_5": precision_at_k(relevance, 5),
        "precision_at_10": precision_at_k(relevance, 10),
        "recall_at_5_within_reviewed_pool": recall_at_k(relevance, 5),
        "recall_at_10_within_reviewed_pool": recall_at_k(relevance, 10),
        "mrr": mean_reciprocal_rank(relevance),
        "ndcg_at_10": ndcg_at_k(relevance, 10),
    }

    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    print("\nTF-IDF PILOT EVALUATION")
    print("=" * 55)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nSaved evaluation report:")
    print(OUTPUT_PATH)

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    main()
