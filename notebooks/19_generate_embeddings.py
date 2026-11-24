import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.services.device_utils import select_torch_device


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data" / "processed" / "job_chunks.parquet"

OUTPUT_DIR = ROOT / "data" / "processed"

EMBEDDINGS_PATH = OUTPUT_DIR / "job_chunk_embeddings.npy"

METADATA_PATH = OUTPUT_DIR / "job_chunk_embedding_metadata.parquet"

MANIFEST_PATH = OUTPUT_DIR / "embedding_manifest.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 16
EXPECTED_DIMENSIONS = 384


def main():

    print("\nFULL JOB EMBEDDING GENERATION")
    print("=" * 60)

    device = select_torch_device()
    print("Selected embedding device:", device)

    # 1. Load the chunk dataset
    chunks = pd.read_parquet(CHUNKS_PATH)

    if chunks.empty:
        raise ValueError("Chunk dataset is empty.")

    # Establish deterministic row ordering.
    chunks = chunks.sort_values(
        ["job_id", "chunk_index"],
        kind="stable"
    ).reset_index(drop=True)

    if not chunks["chunk_id"].is_unique:
        raise ValueError("Duplicate chunk IDs detected.")

    if (chunks["model_input_token_count"] > 256).any():
        raise ValueError("Chunk exceeds model input limit.")

    print("Total chunks:", len(chunks))

    # 2. Load model onto the selected device
    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    actual_device = str(model.device).split(":")[0]
    if actual_device != device:
        raise RuntimeError(
            f"Requested device {device}, but model loaded on "
            f"{model.device}."
        )
    dimensions = model.get_sentence_embedding_dimension()

    if dimensions != EXPECTED_DIMENSIONS:
        raise ValueError("Unexpected embedding dimensions.")

    print("Model:", MODEL_NAME)
    print("Device:", model.device)
    print("Dimensions:", dimensions)
    print("Batch size:", BATCH_SIZE)

    # 3. Prepare text inputs
    texts = chunks["chunk_text"].astype(str).tolist()

    # 4. Generate embeddings in batches
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # 5. Validate generated vectors
    expected_shape = (len(chunks), dimensions)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, received {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid numerical values found.")

    vector_norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(vector_norms, 1.0, atol=1e-4):
        raise ValueError("Embedding normalization failed.")

    print("\nVECTOR VALIDATION")
    print("-" * 60)

    print("Embedding shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Finite values: PASSED")
    print("Normalization: PASSED")
    print("Dimensions: PASSED")

    # 6. Prepare metadata in exactly the same row order
    metadata_columns = [
        "chunk_id",
        "job_id",
        "chunk_index",
        "title",
        "company",
        "location",
        "seniority_level",
        "employment_type",
        "normalized_city",
        "normalized_country",
        "start_token",
        "end_token",
        "content_token_count",
        "model_input_token_count",
        "model_name"
    ]

    metadata = chunks[metadata_columns].copy()

    # 7. Save vectors and their corresponding metadata
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    metadata.to_parquet(
        METADATA_PATH,
        engine="pyarrow",
        index=False
    )

    # 8. Save an embedding manifest
    manifest = {
        "model_name": MODEL_NAME,
        "embedding_dimensions": dimensions,
        "embedding_count": len(embeddings),
        "data_type": "float32",
        "normalized": True,
        "device": str(model.device),
        "batch_size": BATCH_SIZE,
        "chunk_size": 200,
        "chunk_overlap": 40,
        "row_mapping": (
            "Embedding row i corresponds to metadata row i."
        ),
        "embeddings_file": EMBEDDINGS_PATH.name,
        "metadata_file": METADATA_PATH.name
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    # 9. Reload saved artifacts and validate them
    saved_embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    saved_metadata = pd.read_parquet(
        METADATA_PATH
    )

    if saved_embeddings.shape != saved_metadata.shape[:1] + (dimensions,):
        raise ValueError("Saved embedding shape is incorrect.")

    if not saved_metadata["chunk_id"].is_unique:
        raise ValueError("Saved metadata contains duplicate chunk IDs.")

    if not np.array_equal(
        saved_metadata["chunk_id"].to_numpy(),
        chunks["chunk_id"].to_numpy()
    ):
        raise ValueError("Embedding-to-metadata row mapping changed.")

    # 10. Final report
    print("\nFINAL EMBEDDING REPORT")
    print("-" * 60)

    print("Total embeddings:", len(saved_embeddings))
    print("Unique jobs:", saved_metadata["job_id"].nunique())
    print("Vector dimensions:", saved_embeddings.shape[1])
    print("Vector data type:", saved_embeddings.dtype)

    print(
        "Embedding array size:",
        round(saved_embeddings.nbytes / (1024 * 1024), 2),
        "MB"
    )

    print("\nSaved files:")
    print(EMBEDDINGS_PATH)
    print(METADATA_PATH)
    print(MANIFEST_PATH)

    print("\nALL EMBEDDING VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
