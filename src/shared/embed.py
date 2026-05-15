from sentence_transformers import SentenceTransformer


__all__ = ["load_embedder", "embed_texts"]


def load_embedder(model_name: str) -> SentenceTransformer:
    """Load the local SentenceTransformers model named in service config."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    # Normalized vectors make Qdrant cosine scores comparable across queries.
    return model.encode(texts, normalize_embeddings=True).tolist()
