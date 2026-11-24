import numpy as np
import torch

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("\nSENTENCE TRANSFORMER MODEL TEST")
    print("=" * 55)

    # 1. Check Apple GPU availability
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available.")

    device = "mps"

    print("Selected device:", device)

    # 2. Load pretrained model
    print("\nLoading pretrained model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    print("Model loaded successfully.")

    # 3. Inspect model configuration
    print("\nMODEL INFORMATION")
    print("-" * 55)

    print("Model name:", MODEL_NAME)
    print("Model device:", model.device)
    print("Maximum sequence length:", model.max_seq_length)
    print(
        "Embedding dimensions:",
        model.get_sentence_embedding_dimension()
    )

    # 4. Prepare sample sentences
    sentences = [
        "I clean datasets using Python and SQL before analysis.",
        "I preprocess and transform data using Python and SQL.",
        "I develop mobile applications using Swift and iOS."
    ]

    # 5. Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        sentences,
        batch_size=2,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # 6. Validate embedding output
    print("\nEMBEDDING VALIDATION")
    print("-" * 55)

    print("Embedding shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)

    assert embeddings.shape == (3, 384)
    assert np.isfinite(embeddings).all()

    print("All embedding values are finite.")

    norms = np.linalg.norm(embeddings, axis=1)

    print("Vector norms:", np.round(norms, 4))

    assert np.allclose(norms, 1.0, atol=1e-4)

    print("All vectors are normalized.")

    # 7. Compare semantic similarity
    similarity_related = float(
        embeddings[0] @ embeddings[1]
    )

    similarity_unrelated = float(
        embeddings[0] @ embeddings[2]
    )

    print("\nSEMANTIC SIMILARITY TEST")
    print("-" * 55)

    print(
        "Related sentences similarity:",
        round(similarity_related, 4)
    )

    print(
        "Unrelated sentences similarity:",
        round(similarity_unrelated, 4)
    )

    print("\nMODEL TEST PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
