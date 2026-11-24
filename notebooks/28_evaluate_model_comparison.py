import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")

# 1. Load human judgments.
pool = pd.read_csv(BASE / "model_comparison_pool.csv")
blind = pd.read_csv(BASE / "model_comparison_blind_review.csv")
tfidf_labels = pd.read_csv(BASE / "tfidf_top10_for_review.csv")

labels = pd.concat(
    [
        blind[["id", "relevance_grade"]],
        tfidf_labels[["id", "relevance_grade"]],
    ],
    ignore_index=True,
)

labels["relevance_grade"] = pd.to_numeric(
    labels["relevance_grade"], errors="raise"
)

assert labels["id"].is_unique
assert len(labels) == 31
assert set(labels["id"]) == set(pool["id"])

# 2. Load diversified recommendations and construct group mappings.
div = pd.read_csv(
    BASE / "semantic_ranked_jobs_diversified.csv"
)

member_to_group = {}

for _, row in div.iterrows():
    representative = row["id"]
    members = [representative]

    alternatives = row["alternate_job_ids"]

    if pd.notna(alternatives) and str(alternatives).strip():
        members.extend(
            item.strip()
            for item in str(alternatives).split(",")
            if item.strip()
        )

    for member in members:
        if member in member_to_group:
            raise ValueError(
                f"Job {member} belongs to multiple groups."
            )
        member_to_group[member] = representative

labels["group_id"] = labels["id"].map(member_to_group)

if labels["group_id"].isna().any():
    missing = labels.loc[
        labels["group_id"].isna(), "id"
    ].tolist()
    raise ValueError(f"Judged jobs without group mapping: {missing}")

# 3. Assign each group its highest reviewed relevance grade.
group_labels = (
    labels.groupby("group_id")
    .agg(
        relevance_grade=("relevance_grade", "max"),
        judged_member_count=("id", "nunique"),
        judged_member_ids=("id", lambda x: ", ".join(x)),
    )
    .reset_index()
)

judged_groups = set(group_labels["group_id"])

assert len(group_labels) == 27, (
    f"Expected 27 judged groups, got {len(group_labels)}"
)

# 4. Prepare the three model rankings.
tfidf = pd.read_csv(BASE / "tfidf_ranked_jobs.csv")
semantic = pd.read_csv(BASE / "semantic_ranked_jobs.csv")

tfidf["original_rank"] = np.arange(1, len(tfidf) + 1)

rankings = {
    "TF-IDF": tfidf.rename(
        columns={"id": "job_id"}
    )[["job_id", "original_rank"]].rename(
        columns={"original_rank": "rank"}
    ),

    "Semantic": semantic.rename(
        columns={
            "job_id": "job_id",
            "semantic_rank": "rank",
        }
    )[["job_id", "rank"]],

    "Diversified Semantic": div[
        ["id", "diversified_rank"]
    ].rename(
        columns={
            "id": "job_id",
            "diversified_rank": "rank",
        }
    ),
}

def dcg(grades):
    grades = np.asarray(grades, dtype=float)
    discounts = np.log2(np.arange(2, len(grades) + 2))
    return float(((2 ** grades - 1) / discounts).sum())


# 5. Evaluate each model on the same 27 recommendation groups.
def evaluate(name, ranking):
    ranking = ranking.copy()

    ranking["rank"] = pd.to_numeric(
        ranking["rank"], errors="raise"
    )

    ranking["group_id"] = ranking["job_id"].map(
        lambda job_id: member_to_group.get(job_id)
    )

    ranking = ranking[
        ranking["group_id"].isin(judged_groups)
    ]

    ranking = (
        ranking.sort_values("rank")
        .drop_duplicates("group_id")
        .reset_index(drop=True)
    )

    ranking["evaluation_rank"] = np.arange(
        1, len(ranking) + 1
    )

    ranking = ranking.merge(
        group_labels,
        on="group_id",
        how="left",
        validate="one_to_one",
    )

    ranking["relevant"] = ranking["relevance_grade"] >= 1

    assert len(ranking) == 27, (
        f"{name}: expected 27 groups, got {len(ranking)}"
    )

    top5 = ranking.head(5)
    top10 = ranking.head(10)

    relevant_count = int(ranking["relevant"].sum())

    relevant_ranks = ranking.loc[
        ranking["relevant"], "evaluation_rank"
    ].tolist()

    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    actual_dcg = dcg(top10["relevance_grade"].tolist())

    ideal_grades = sorted(
        ranking["relevance_grade"].tolist(),
        reverse=True,
    )[:10]

    ideal_dcg = dcg(ideal_grades)

    metrics = {
        "model": name,
        "judged_recommendation_groups": len(ranking),
        "relevant_groups": relevant_count,
        "precision_at_5": top5["relevant"].mean(),
        "precision_at_10": top10["relevant"].mean(),
        "recall_at_10_within_judged_groups": (
            top10["relevant"].sum() / relevant_count
            if relevant_count else 0.0
        ),
        "mrr": mrr,
        "ndcg_at_10": (
            actual_dcg / ideal_dcg if ideal_dcg else 0.0
        ),
    }

    return metrics, ranking


results = []
details = []

for name, ranking in rankings.items():
    metrics, detail = evaluate(name, ranking)
    results.append(metrics)

    detail.insert(0, "model", name)
    details.append(detail)

metrics_df = pd.DataFrame(results)
details_df = pd.concat(details, ignore_index=True)

# 6. Add readable representative job information.
representative_info = div[
    ["id", "title", "company", "location"]
].rename(columns={
    "id": "group_id",
    "title": "representative_title",
    "company": "representative_company",
    "location": "representative_location",
})

details_df = details_df.merge(
    representative_info,
    on="group_id",
    how="left",
    validate="many_to_one",
)

# 7. Save evaluation artifacts.
metrics_df.to_csv(
    BASE / "model_comparison_metrics.csv",
    index=False,
)

details_df.to_csv(
    BASE / "model_comparison_ranked_judgments.csv",
    index=False,
)

group_labels.to_csv(
    BASE / "model_comparison_group_mapping.csv",
    index=False,
)

# 8. Print results.
print("\nGROUP-LEVEL MODEL COMPARISON")
print("=" * 100)
print(metrics_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))

print("\nTOP 10 RECOMMENDATION GROUPS BY MODEL")
print("=" * 100)

for name in rankings:
    print(f"\n{name}")

    subset = details_df[
        details_df["model"] == name
    ].sort_values("evaluation_rank").head(10)

    for _, row in subset.iterrows():
        print(
            f"Rank {int(row['evaluation_rank']):>2} | "
            f"Grade {int(row['relevance_grade'])} | "
            f"{row['representative_title']} | "
            f"{row['representative_company']}"
        )

print("\nEVALUATION VALIDATION")
print("Original judged postings:", len(labels))
print("Unique judged recommendation groups:", len(group_labels))
print("Relevant groups:", int(
    (group_labels["relevance_grade"] >= 1).sum()
))

print("\nSAVED:")
print("model_comparison_metrics.csv")
print("model_comparison_ranked_judgments.csv")
print("model_comparison_group_mapping.csv")

print("\nLIMITATIONS:")
print("- One synthetic candidate profile.")
print("- 31 manually judged postings grouped into 27 recommendations.")
print("- Group relevance uses the maximum grade of reviewed members.")
print("- Recall is limited to relevant groups in this judged pool.")
print("- Preliminary offline evaluation, not full-corpus recall.")
