from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "jobs_cleaned.parquet"

OUTPUT_PATH = ROOT / "data" / "processed" / "token_length_audit.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("\nTOKEN LENGTH AUDIT")
    print("=" * 55)

    # Load eligible job records
    jobs = pd.read_parquet(DATA_PATH)

    jobs = jobs[
        jobs["eligible_for_semantic_matching"].eq(True)
    ].copy()

    # Load the exact tokenizer used by our embedding model
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    tokenizer = model.tokenizer

    max_tokens = model.max_seq_length

    print("Model:", MODEL_NAME)
    print("Maximum sequence length:", max_tokens)
    print("Eligible jobs:", len(jobs))

    # Include job titles because they will be part of
    # our final embedding documents.
    texts = (
        "Job title: "
        + jobs["title"].fillna("").astype(str)
        + "\nJob description: "
        + jobs["description"].fillna("").astype(str)
    ).tolist()

    # Tokenize without truncation
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False
    )

    token_lengths = [
        len(input_ids)
        for input_ids in encoded["input_ids"]
    ]

    # Add audit results
    audit = jobs[["id", "title"]].copy()

    audit["token_count"] = token_lengths

    audit["exceeds_model_limit"] = (
        audit["token_count"] > max_tokens
    )

    # Calculate statistics
    print("\nTOKEN COUNT STATISTICS")
    print("-" * 55)

    print(
        audit["token_count"]
        .describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.99])
        .round(2)
        .to_string()
    )

    exceeds = audit["exceeds_model_limit"].sum()

    print("\nJOBS EXCEEDING MODEL LIMIT:", int(exceeds))

    print(
        "Percentage exceeding limit:",
        round(exceeds / len(audit) * 100, 2),
        "%"
    )

    print("\nFIVE LONGEST JOB DESCRIPTIONS")
    print("-" * 55)

    print(
        audit.nlargest(5, "token_count")[
            ["title", "token_count"]
        ].to_string(index=False)
    )

    print("\nFIVE SHORTEST JOB DESCRIPTIONS")
    print("-" * 55)

    print(
        audit.nsmallest(5, "token_count")[
            ["title", "token_count"]
        ].to_string(index=False)
    )

    # Save token-length audit
    audit.to_csv(OUTPUT_PATH, index=False)

    print("\nAUDIT SAVED TO:")
    print(OUTPUT_PATH)

    print("\nTOKEN LENGTH AUDIT COMPLETED")


if __name__ == "__main__":
    main()
