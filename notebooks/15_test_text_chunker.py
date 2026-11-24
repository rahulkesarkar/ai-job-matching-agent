from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from app.services.text_chunker import chunk_text


ROOT = Path(__file__).resolve().parents[1]

JOBS_PATH = ROOT / "data" / "processed" / "jobs_cleaned.parquet"

AUDIT_PATH = ROOT / "data" / "processed" / "token_length_audit.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("\nTEXT CHUNKER VALIDATION")
    print("=" * 60)

    # Load the exact tokenizer used by our embedding model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    jobs = pd.read_parquet(JOBS_PATH)

    eligible = jobs[
        jobs["eligible_for_semantic_matching"].eq(True)
    ].copy()

    audit = pd.read_csv(AUDIT_PATH)

    # Select the longest, median and shortest eligible records
    longest_id = audit.loc[audit["token_count"].idxmax(), "id"]

    median_id = audit.iloc[
        (audit["token_count"] - audit["token_count"].median()).abs().argsort()[:1]
    ]["id"].iloc[0]

    shortest_id = audit.loc[audit["token_count"].idxmin(), "id"]

    selected_ids = [longest_id, median_id, shortest_id]

    selected_jobs = eligible[
        eligible["id"].isin(selected_ids)
    ].copy()

    if len(selected_jobs) != 3:
        raise ValueError("Could not locate all three test jobs.")

    for _, job in selected_jobs.iterrows():

        document = (
            f"Job title: {job['title']}\n"
            f"Job description: {job['description']}"
        )

        original_tokens = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(document)
        )

        chunks = chunk_text(
            text=document,
            tokenizer=tokenizer,
            max_seq_length=256,
            chunk_size=200,
            overlap=40
        )

        if not chunks:
            raise ValueError(f"No chunks generated for {job['id']}")

        if chunks[0]["start_token"] != 0:
            raise ValueError("Chunking does not start at token zero.")

        if chunks[-1]["end_token"] != len(original_tokens):
            raise ValueError("The final chunk does not cover the full document.")

        for chunk in chunks:
            if chunk["model_input_token_count"] > 256:
                raise ValueError("A chunk exceeds the model input limit.")

        for previous, current in zip(chunks, chunks[1:]):
            if current["start_token"] != previous["end_token"] - 40:
                raise ValueError("Expected 40-token overlap was not preserved.")

        print("\nJOB:", job["title"])
        print("Original token count:", len(original_tokens))
        print("Number of chunks:", len(chunks))
        print("First chunk tokens:", chunks[0]["content_token_count"])
        print("Last chunk tokens:", chunks[-1]["content_token_count"])
        print(
            "Largest model input:",
            max(c["model_input_token_count"] for c in chunks)
        )
        print("Full document coverage: PASSED")
        print("Overlap validation: PASSED")
        print("Model input limit: PASSED")

    print("\nALL CHUNKING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
