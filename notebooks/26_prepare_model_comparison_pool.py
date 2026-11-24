from pathlib import Path
import pandas as pd

BASE = Path("data/processed")

tfidf = pd.read_csv(BASE / "tfidf_ranked_jobs.csv")
semantic = pd.read_csv(BASE / "semantic_ranked_jobs.csv")
diversified = pd.read_csv(BASE / "semantic_ranked_jobs_diversified.csv")
old_labels = pd.read_csv(BASE / "tfidf_top10_for_review.csv")
jobs = pd.read_parquet(BASE / "jobs_cleaned.parquet")

jobs["id"] = jobs["id"].astype(str)

# Select top 10 from each ranking.
tfidf_top = tfidf.head(10).copy()
tfidf_top["id"] = tfidf_top["id"].astype(str)

semantic_top = semantic.head(10).copy()
semantic_top["id"] = semantic_top["job_id"].astype(str)

diversified_top = diversified.head(10).copy()
diversified_top["id"] = diversified_top["job_id"].astype(str)

# Collect ranked job IDs.
selected_ids = set(tfidf_top["id"]) | set(semantic_top["id"]) | set(diversified_top["id"])

# Select 10 additional random jobs.
eligible = jobs.dropna(subset=["description"]).copy()
eligible = eligible[eligible["description"].str.strip().ne("")]
eligible = eligible[~eligible["id"].isin(selected_ids)]

random_jobs = eligible.sample(n=10, random_state=42)
random_ids = set(random_jobs["id"])

# Build pool using the cleaned dataset as the source of truth.
pool_ids = selected_ids | random_ids

pool = jobs.loc[
    jobs["id"].isin(pool_ids),
    ["id", "title", "company", "location", "description"]
].drop_duplicates("id").copy()

# Add ranking membership and positions.
pool = pool.merge(
    tfidf_top[["id"]].assign(tfidf_top10=True),
    on="id", how="left"
)
pool = pool.merge(
    semantic_top[["id"]].assign(semantic_top10=True),
    on="id", how="left"
)
pool = pool.merge(
    diversified_top[["id"]].assign(diversified_semantic_top10=True),
    on="id", how="left"
)

for column in ["tfidf_top10", "semantic_top10", "diversified_semantic_top10"]:
    pool[column] = pool[column].fillna(False).astype(bool)

pool = pool.merge(
    tfidf_top[["id"]].assign(tfidf_rank=range(1, len(tfidf_top) + 1)),
    on="id", how="left"
)
pool = pool.merge(
    semantic_top[["id"]].assign(semantic_rank=range(1, len(semantic_top) + 1)),
    on="id", how="left"
)
pool = pool.merge(
    diversified_top[["id"]].assign(
        diversified_rank=range(1, len(diversified_top) + 1)
    ),
    on="id", how="left"
)

# Carry forward existing relevance judgments.
labels = old_labels[
    ["id", "relevance_grade", "judgment_reason", "judgment_status"]
].copy()
labels["id"] = labels["id"].astype(str)

pool = pool.merge(labels, on="id", how="left")
pool["random_sample"] = pool["id"].isin(random_ids)
pool["relevance_grade"] = pd.to_numeric(
    pool["relevance_grade"], errors="coerce"
)

pool = pool.sort_values(
    ["tfidf_rank", "semantic_rank", "diversified_rank"],
    na_position="last"
)

output = BASE / "model_comparison_pool.csv"
pool.to_csv(output, index=False)

print("\nMODEL COMPARISON POOL")
print("---------------------")
print(f"Unique jobs: {len(pool)}")
print(f"Already judged: {pool['relevance_grade'].notna().sum()}")
print(f"Awaiting judgment: {pool['relevance_grade'].isna().sum()}")
print(f"Random sample: {pool['random_sample'].sum()}")
print(f"Missing titles: {pool['title'].isna().sum()}")
print(f"Missing companies: {pool['company'].isna().sum()}")
print(f"Missing descriptions: {pool['description'].isna().sum()}")
print(f"\nSaved: {output}")

print("\nUNJUDGED JOBS")
print(pool.loc[
    pool["relevance_grade"].isna(),
    ["id", "title", "company", "location",
     "tfidf_rank", "semantic_rank", "diversified_rank",
     "random_sample"]
].to_string(index=False))

assert pool["id"].is_unique, "Duplicate job IDs detected"
assert pool["description"].notna().all(), "Missing descriptions detected"
