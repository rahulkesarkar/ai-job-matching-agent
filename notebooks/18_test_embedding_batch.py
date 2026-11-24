from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data" / "processed" / "job_chunks.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 16
TEST_SIZE = 64


def main():

    print("\nEMBEDDING BATCH TEST")
    print("=" * 55)

    # 1. Verify GPU availability
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available.")

    # 2. Load chunk dataset
    chunks = pd.read_parquet(CHUNKS_PATH)

    sample = chunks.head(TEST_SIZE).copy()

    if len(sample) != TEST_SIZE:
        raise ValueError("Expected 64 test chunks.")

    # 3. Load model directly onto Apple GPU
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="mps"
    )

    if not str(model.device).startswith("mps"):
        raise RuntimeError("Model is not running on MPS.")

    print("Model device:", model.device)
    print("Embedding dimensions:",
          model.get_sentence_embedding_dimension())
    print("Maximum sequence length:", model.max_seq_length)

    # 4. Check input lengths
    if (sample["model_input_token_count"] > model.max_seq_length).any():
        raise ValueError("Input exceeds model token limit.")

    texts = sample["chunk_text"].tolist()

    # 5. Generate embeddings
    print("\nGenerating test embeddings...")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    elapsed = time.perf_counter() - start_time

    # 6. Validate output
    print("\nEMBEDDING VALIDATION")
    print("-" * 55)

    print("Input chunks:", len(texts))
    print("Embedding shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Generation time:", round(elapsed, 2), "seconds")

    expected_shape = (
        TEST_SIZE,
        model.get_sentence_embedding_dimension()
    )

    if embeddings.shape != expected_shape:
        raise ValueError("Unexpected embedding dimensions.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid numerical values detected.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embedding normalization failed.")

    print("All values finite: PASSED")
    print("Vector normalization: PASSED")
    print("Embedding dimensions: PASSED")
    print("Apple MPS execution: VERIFIED")

    print("\nEMBEDDING BATCH TEST PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
