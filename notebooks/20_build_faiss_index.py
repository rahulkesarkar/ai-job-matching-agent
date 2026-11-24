import json
from pathlib import Path
from importlib.metadata import version

import faiss
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"

EMBEDDINGS_PATH = DATA_DIR / "job_chunk_embeddings.npy"
METADATA_PATH = DATA_DIR / "job_chunk_embedding_metadata.parquet"
INDEX_PATH = DATA_DIR / "faiss_job_chunks.index"
INDEX_MANIFEST_PATH = DATA_DIR / "faiss_index_manifest.json"


def main():

    print("\nFAISS VECTOR INDEX BUILD")
    print("=" * 60)

    # 1. Load embeddings and metadata
    embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    metadata = pd.read_parquet(METADATA_PATH)

    # 2. Validate inputs
    if embeddings.dtype != np.float32:
        raise ValueError("Expected float32 embeddings.")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts differ.")

    if not metadata["chunk_id"].is_unique:
        raise ValueError("Duplicate chunk IDs detected.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values detected.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not normalized.")

    print("Embeddings loaded:", len(embeddings))
    print("Dimensions:", embeddings.shape[1])
    print("Unique jobs:", metadata["job_id"].nunique())

    # 3. Create an exact inner-product index
    dimensions = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimensions)

    # FAISS expects contiguous float32 vectors.
    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    # 4. Add all vectors to the index
    print("\nAdding vectors to FAISS...")

    index.add(embeddings)

    if index.ntotal != len(embeddings):
        raise ValueError("Not all vectors were indexed.")

    # 5. Save the index
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # 6. Reload and validate the saved index
    loaded_index = faiss.read_index(
        str(INDEX_PATH)
    )

    if loaded_index.ntotal != len(metadata):
        raise ValueError("Saved index vector count is incorrect.")

    if loaded_index.d != dimensions:
        raise ValueError("Saved index dimensions are incorrect.")

    # 7. Test a real vector search
    scores, indices = loaded_index.search(
        embeddings[0:1],
        5
    )

    if not np.isclose(scores[0][0], 1.0, atol=1e-4):
        raise ValueError("Self-similarity search validation failed.")

    print("\nSAMPLE VECTOR SEARCH")
    print("-" * 60)

    for position, (vector_id, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):
        job = metadata.iloc[int(vector_id)]

        print(
            f"{position}. {job['title']} | "
            f"{job['company']} | "
            f"Similarity: {score:.4f}"
        )

    # 8. Save index configuration
    manifest = {
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "faiss_version": version("faiss-cpu"),
        "vector_count": loaded_index.ntotal,
        "embedding_dimensions": loaded_index.d,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vectors_normalized": True,
        "exact_search": True,
        "metadata_file": METADATA_PATH.name,
        "embeddings_file": EMBEDDINGS_PATH.name,
        "index_file": INDEX_PATH.name,
        "mapping": "FAISS vector ID equals metadata row position"
    }

    INDEX_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8"
    )

    print("\nINDEX VALIDATION")
    print("-" * 60)

    print("FAISS version:", version("faiss-cpu"))
    print("Index type: IndexFlatIP")
    print("Indexed vectors:", loaded_index.ntotal)
    print("Vector dimensions:", loaded_index.d)
    print("Index file size:",
          round(INDEX_PATH.stat().st_size / (1024 * 1024), 2),
          "MB")

    print("\nIndex saved:", INDEX_PATH)
    print("Manifest saved:", INDEX_MANIFEST_PATH)

    print("\nALL FAISS VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
