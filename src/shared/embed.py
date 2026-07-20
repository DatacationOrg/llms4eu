from sentence_transformers import SentenceTransformer


__all__ = ["load_embedder", "embed_texts"]


def load_embedder(
    model_name: str, *, local_files_only: bool = True
) -> SentenceTransformer:
    """Load a SentenceTransformers model from the local cache by default.

    Use local_files_only=False only for explicit model download/cache warmup.
    """
    return SentenceTransformer(model_name, local_files_only=local_files_only)


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    prompt_name: str | None = None,
    batch_size: int | None = None,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    # Normalized vectors make Qdrant cosine scores comparable across queries.
    # prompt_name can be e.g. "query" for instruction-aware models like Qwen3-Embedding.
    kwargs: dict = {
        "normalize_embeddings": True,
        "show_progress_bar": show_progress_bar,
    }
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return model.encode(texts, **kwargs).tolist()
