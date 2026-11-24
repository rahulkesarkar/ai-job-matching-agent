import math
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

AUDIT_PATH = ROOT / "data" / "processed" / "token_length_audit.csv"
JOBS_PATH = ROOT / "data" / "processed" / "jobs_cleaned.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "chunk_volume_estimate.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 200
OVERLAP = 40
EMBEDDING_DIMENSIONS = 384


def calculate_chunk_count(token_count, chunk_size, overlap):

    if token_count <= 0:
        return 0

    if token_count <= chunk_size:
        return 1

    stride = chunk_size - overlap

    return 1 + math.ceil(
        (token_count - chunk_size) / stride
    )


def main():

    print("\nCHUNK VOLUME ESTIMATION")
    print("=" * 55)

    audit = pd.read_csv(AUDIT_PATH)
    jobs = pd.read_parquet(JOBS_PATH)

    eligible_jobs = jobs[
        jobs["eligible_for_semantic_matching"].eq(True)
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    special_tokens = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    if audit["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in token audit.")

    if set(audit["id"]) != set(eligible_jobs["id"]):
        raise ValueError(
            "Token audit does not match eligible job records."
        )

    # The original audit included special tokens.
    # Remove them to calculate content chunk sizes.
    audit["content_tokens"] = (
        audit["token_count"] - special_tokens
    )

    if (audit["content_tokens"] < 0).any():
        raise ValueError("Invalid token counts found.")

    audit["estimated_chunks"] = audit["content_tokens"].apply(
        lambda count: calculate_chunk_count(
            count,
            CHUNK_SIZE,
            OVERLAP
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit[
        ["id", "title", "content_tokens", "estimated_chunks"]
    ].to_csv(OUTPUT_PATH, index=False)

    total_chunks = int(audit["estimated_chunks"].sum())

    estimated_storage = (
        total_chunks
        * EMBEDDING_DIMENSIONS
        * 4
    )

    print("Eligible jobs:", len(audit))
    print("Special tokens per document:", special_tokens)
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap:", OVERLAP)
    print("Stride:", CHUNK_SIZE - OVERLAP)

    print("\nTOKEN DISTRIBUTION")
    print(
        audit["content_tokens"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nCHUNK WORKLOAD")
    print("Total estimated chunks:", total_chunks)
    print(
        "Average chunks per job:",
        round(audit["estimated_chunks"].mean(), 2)
    )
    print(
        "Maximum chunks for one job:",
        audit["estimated_chunks"].max()
    )

    print("\nEMBEDDING STORAGE ESTIMATE")
    print(
        "Float32 embedding storage:",
        round(estimated_storage / (1024 * 1024), 2),
        "MB"
    )

    print("\nSaved workload report:")
    print(OUTPUT_PATH)

    print("\nCHUNK WORKLOAD ESTIMATION COMPLETED")


if __name__ == "__main__":
    main()
