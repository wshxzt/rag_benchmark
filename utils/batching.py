"""
Dynamic token-aware batching for text embedding API calls.

text-embedding-004 limits: 2048 tokens per input, 20000 tokens per batch.
Strategy: estimate tokens as len(text) / 4, accumulate until approaching
the per-batch limit, then flush. This avoids fixed batch sizes that fail
on long documents.
"""

CHARS_PER_TOKEN      = 4
MAX_TOKENS_PER_BATCH = 18000  # safely under the 20k token limit per batch
MAX_ITEMS_PER_BATCH  = 250    # hard API limit: 250 instances per prediction call


def dynamic_batches(texts: list[str]) -> list[list[int]]:
    """
    Returns lists of indices into `texts`, grouped so that each batch stays
    under both MAX_TOKENS_PER_BATCH and MAX_ITEMS_PER_BATCH.

    Example:
        for idx_batch in dynamic_batches(texts):
            batch_texts = [texts[i] for i in idx_batch]
            embeddings  = model.get_embeddings(batch_texts)
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for i, text in enumerate(texts):
        estimated = len(text) / CHARS_PER_TOKEN
        if current and (
            current_tokens + estimated > MAX_TOKENS_PER_BATCH
            or len(current) >= MAX_ITEMS_PER_BATCH
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(i)
        current_tokens += estimated

    if current:
        batches.append(current)

    return batches
