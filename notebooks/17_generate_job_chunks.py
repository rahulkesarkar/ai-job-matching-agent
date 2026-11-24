from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from app.services.text_chunker import chunk_text


ROOT = Path(__file__).resolve().parents[1]

JOBS_PATH = ROOT / "data" / "processed" / "jobs_cleaned.parquet"

ESTIMATE_PATH = ROOT / "data" / "processed" / "chunk_volume_estimate.csv"

OUTPUT_PATH = ROOT / "data" / "processed" / "job_chunks.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SEQ_LENGTH = 256
CHUNK_SIZE = 200
OVERLAP = 40


def main():

    print("\nJOB CHUNK GENERATION")
    print("=" * 60)

    # 1. Load eligible job records
    jobs = pd.read_parquet(JOBS_PATH)

    jobs = jobs[
        jobs["eligible_for_semantic_matching"].eq(True)
    ].copy()

    print("Eligible jobs loaded:", len(jobs))

    # 2. Load the selected model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Model:", MODEL_NAME)
    print("Maximum input tokens:", MAX_SEQ_LENGTH)

    # 3. Generate chunks for each job
    chunk_records = []

    for _, job in jobs.iterrows():

        document = (
            f"Job title: {job['title']}\n"
            f"Job description: {job['description']}"
        )

        chunks = chunk_text(
            text=document,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        if not chunks:
            raise ValueError(
                f"No chunks generated for job {job['id']}"
            )

        # Every chunk retains its parent job information.
        for chunk in chunks:

            chunk_records.append({
                "job_id": job["id"],
                "chunk_id": f"{job['id']}_chunk_{chunk['chunk_index']:03d}",
                "chunk_index": chunk["chunk_index"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "seniority_level": job["seniority_level"],
                "employment_type": job["employment_type"],
                "normalized_city": job["normalized_city"],
                "normalized_country": job["normalized_country"],
                "start_token": chunk["start_token"],
                "end_token": chunk["end_token"],
                "content_token_count": chunk["content_token_count"],
                "model_input_token_count": chunk["model_input_token_count"],
                "chunk_text": chunk["text"],
                "model_name": MODEL_NAME
            })

    # 4. Convert generated records into a DataFrame
    chunk_df = pd.DataFrame(chunk_records)

    # 5. Validate generated chunks
    if chunk_df.empty:
        raise ValueError("No chunks were generated.")

    if not chunk_df["chunk_id"].is_unique:
        raise ValueError("Duplicate chunk IDs found.")

    generated_job_ids = set(chunk_df["job_id"])
    expected_job_ids = set(jobs["id"])

    if generated_job_ids != expected_job_ids:
        raise ValueError("Some eligible jobs are missing from the chunks.")

    if (chunk_df["model_input_token_count"] > MAX_SEQ_LENGTH).any():
        raise ValueError("Some chunks exceed the model input limit.")

    # 6. Validate token coverage and overlap for every job
    for job_id, group in chunk_df.groupby("job_id"):

        group = group.sort_values("chunk_index")

        if group.iloc[0]["start_token"] != 0:
            raise ValueError(f"Missing initial tokens: {job_id}")

        for previous, current in zip(
            group.iloc[:-1].itertuples(),
            group.iloc[1:].itertuples()
        ):

            if current.start_token != previous.end_token - OVERLAP:
                raise ValueError(
                    f"Incorrect overlap in job {job_id}"
                )

    # 7. Save chunk dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    chunk_df.to_parquet(
        OUTPUT_PATH,
        engine="pyarrow",
        index=False
    )

    # 8. Compare actual chunk count with our estimate
    estimated = pd.read_csv(ESTIMATE_PATH)

    estimated_total = int(
        estimated["estimated_chunks"].sum()
    )

    actual_total = len(chunk_df)

    # 9. Print generation report
    print("\nGENERATION REPORT")
    print("-" * 60)

    print("Jobs processed:", len(jobs))
    print("Estimated chunks:", estimated_total)
    print("Actual chunks:", actual_total)

    print(
        "Average chunks per job:",
        round(actual_total / len(jobs), 2)
    )

    print(
        "Maximum chunks for one job:",
        int(chunk_df.groupby("job_id").size().max())
    )

    print(
        "Largest model input:",
        int(chunk_df["model_input_token_count"].max())
    )

    print("Unique chunk IDs:", chunk_df["chunk_id"].nunique())

    if actual_total == estimated_total:
        print("Chunk estimate comparison: MATCH")
    else:
        print("Chunk estimate comparison: DIFFERENT")
        print("Investigate tokenizer and estimation differences.")

    print("\nProcessed chunk file:")
    print(OUTPUT_PATH)

    print("\nALL GENERATION VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
