from langchain_ollama import ChatOllama
from pydantic import BaseModel


__all__ = ["structured_local_model"]


def structured_local_model(model_id: str, schema: type[BaseModel]):
    """Create a local Ollama chat model that returns the requested Pydantic shape."""
    return ChatOllama(model=model_id, temperature=0).with_structured_output(
        schema,
        method="json_schema",
    )
