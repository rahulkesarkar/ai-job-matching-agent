from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

CANDIDATE_PATH = (
    ROOT / "data" / "sample" / "sample_candidate_profile.txt"
)

METADATA_PATH = DATA_DIR / "job_chunk_embedding_metadata.parquet"

INDEX_PATH = DATA_DIR / "faiss_job_chunks.index"

OUTPUT_PATH = DATA_DIR / "semantic_ranked_jobs.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EXPECTED_DIMENSIONS = 384


def main():

    print("\nSEMANTIC JOB RETRIEVAL")
    print("=" * 60)

    # 1. Load candidate profile
    candidate_raw = CANDIDATE_PATH.read_text(
        encoding="utf-8"
    )

    if "Target Roles:" not in candidate_raw:
        raise ValueError("Candidate profile has no Target Roles section.")

    # Exclude career preferences from semantic text.
    candidate_text = candidate_raw.split(
        "Target Roles:", 1
    )[1]

    candidate_text = candidate_text.split(
        "Career Preferences:", 1
    )[0]

    candidate_text = "Target Roles: " + candidate_text.strip()

    if not candidate_text.strip():
        raise ValueError("Candidate semantic profile is empty.")

    # 2. Load chunk metadata and FAISS index
    metadata = pd.read_parquet(METADATA_PATH)

    index = faiss.read_index(str(INDEX_PATH))

    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata counts differ.")

    if index.d != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected FAISS embedding dimensions.")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Expected an inner-product FAISS index.")

    if not metadata["chunk_id"].is_unique:
        raise ValueError("Duplicate chunk IDs in metadata.")

    print("Candidate profile loaded.")
    print("Indexed chunks:", index.ntotal)
    print("Unique jobs:", metadata["job_id"].nunique())

    # 3. Load embedding model onto Apple GPU
    print("\nLoading Sentence Transformer...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="mps"
    )

    print("Model device:", model.device)

    # 4. Generate candidate embedding
    candidate_embedding = model.encode(
        [candidate_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    candidate_embedding = np.ascontiguousarray(
        candidate_embedding,
        dtype=np.float32
    )

    if candidate_embedding.shape != (1, EXPECTED_DIMENSIONS):
        raise ValueError("Candidate embedding shape is incorrect.")

    if not np.allclose(
        np.linalg.norm(candidate_embedding),
        1.0,
        atol=1e-4
    ):
        raise ValueError("Candidate embedding is not normalized.")

    # 5. Search every indexed chunk using exact similarity search
    print("\nSearching FAISS index...")

    scores, indices = index.search(
        candidate_embedding,
        index.ntotal
    )

    valid_mask = indices[0] >= 0

    vector_ids = indices[0][valid_mask]
    similarity_scores = scores[0][valid_mask]

    if len(vector_ids) != len(metadata):
        raise ValueError("Search did not return every indexed chunk.")

    # 6. Map vector IDs back to their metadata
    hits = metadata.iloc[vector_ids].copy().reset_index(drop=True)

    hits["semantic_similarity"] = similarity_scores

    # 7. Aggregate chunk-level results into one result per job
    best_indices = (
        hits.groupby("job_id")["semantic_similarity"].idxmax()
    )

    ranked_jobs = hits.loc[best_indices].copy()

    ranked_jobs = ranked_jobs.sort_values(
        "semantic_similarity",
        ascending=False
    ).reset_index(drop=True)

    if ranked_jobs["job_id"].duplicated().any():
        raise ValueError("Duplicate jobs remain after aggregation.")

    # 8. Add final ranking positions
    ranked_jobs.insert(
        0,
        "semantic_rank",
        range(1, len(ranked_jobs) + 1)
    )

    output_columns = [
        "semantic_rank",
        "job_id",
        "title",
        "company",
        "location",
        "seniority_level",
        "employment_type",
        "normalized_city",
        "normalized_country",
        "semantic_similarity",
        "chunk_id",
        "chunk_index"
    ]

    ranked_jobs[output_columns].to_csv(
        OUTPUT_PATH,
        index=False
    )

    # 9. Display top 10 unique jobs
    print("\nTOP 10 SEMANTIC JOB MATCHES")
    print("=" * 60)

    print(
        ranked_jobs[output_columns]
        .head(10)
        .round({"semantic_similarity": 4})
        .to_string(index=False)
    )

    print("\nRETRIEVAL INFORMATION")
    print("-" * 60)

    print("Candidate embedding dimensions:", candidate_embedding.shape[1])
    print("Chunks searched:", len(vector_ids))
    print("Unique jobs ranked:", len(ranked_jobs))
    print("Aggregation method: Maximum chunk similarity")
    print("Similarity metric: Cosine similarity")

    print("\nComplete ranking saved to:")
    print(OUTPUT_PATH)

    print("\nSEMANTIC RETRIEVAL COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
