from typing import Any


def chunk_text(
    text: str,
    tokenizer: Any,
    max_seq_length: int = 256,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:

    """
    Split long text into overlapping, tokenizer-based chunks.

    Parameters
    ----------
    text:
        Original text to split.

    tokenizer:
        Tokenizer belonging to the embedding model.

    max_seq_length:
        Maximum input length accepted by the model.

    chunk_size:
        Maximum content tokens in each chunk.

    overlap:
        Number of shared tokens between consecutive chunks.

    Returns
    -------
    List of dictionaries containing chunk text and metadata.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if max_seq_length <= 0 or chunk_size <= 0:
        raise ValueError("Token limits must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    # Identify tokens reserved for model-specific special tokens.
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    available_tokens = max_seq_length - special_tokens

    if chunk_size > available_tokens:
        raise ValueError(
            "Chunk size exceeds available model token capacity."
        )

    # Convert the complete document into token IDs.
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    if not token_ids:
        return []

    stride = chunk_size - overlap

    chunks = []

    start = 0

    while start < len(token_ids):

        end = min(start + chunk_size, len(token_ids))

        current_tokens = token_ids[start:end]

        # Convert token IDs back into readable text.
        chunk_content = tokenizer.decode(
            current_tokens,
            skip_special_tokens=True
        )

        # Verify the actual encoded input length.
        model_input_tokens = len(
            tokenizer.encode(
                chunk_content,
                add_special_tokens=True
            )
        )

        if model_input_tokens > max_seq_length:
            raise ValueError(
                f"Chunk exceeds model limit: {model_input_tokens}"
            )

        chunks.append({
            "chunk_index": len(chunks),
            "start_token": start,
            "end_token": end,
            "content_token_count": len(current_tokens),
            "model_input_token_count": model_input_tokens,
            "text": chunk_content
        })

        if end == len(token_ids):
            break

        start += stride

    return chunks
