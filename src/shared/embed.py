import torch
from sentence_transformers import SentenceTransformer


__all__ = ["load_embedder", "embed_texts"]


def load_embedder(
    model_name: str,
    *,
    local_files_only: bool = True,
    dtype: str | None = None,
    attn_implementation: str | None = None,
    revision: str | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformers model from the local cache by default.

    Use local_files_only=False only for explicit model download/cache warmup.

    `revision` pins a specific cached snapshot. A cache can hold several
    revisions of one model where only some carry weights, and `refs/main` may
    point at a metadata-only one; loading then fails with "does not appear to
    have a file named model.safetensors" despite the weights being on disk.
    """
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["dtype"] = getattr(torch, dtype)
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    return SentenceTransformer(
        model_name,
        local_files_only=local_files_only,
        model_kwargs=model_kwargs,
        revision=revision,
    )


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
