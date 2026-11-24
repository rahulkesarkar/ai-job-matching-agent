import pandas as pd

source = "data/processed/model_comparison_pool.csv"
output = "data/processed/model_comparison_blind_review.csv"

df = pd.read_csv(source)
review = df[df["relevance_grade"].isna()].copy()

review = review.sample(frac=1, random_state=42)

review = review[[
    "id", "title", "company", "location", "description"
]]

review["relevance_grade"] = ""
review["judgment_reason"] = ""

review.to_csv(output, index=False)

print("Blind review file created.")
print("Jobs to review:", len(review))
print("File:", output)
print("\nReview order:")
print(review[["id", "title", "company", "location"]].to_string(index=False))
