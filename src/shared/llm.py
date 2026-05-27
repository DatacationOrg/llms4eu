from langchain_ollama import ChatOllama
from pydantic import BaseModel


__all__ = ["structured_local_model"]


def structured_local_model(
    model_id: str,
    schema: type[BaseModel],
    reasoning: bool | str | None = True,
    num_ctx: int | None = None,
    num_predict: int | None = None,
):
    """Create a local Ollama chat model that returns the requested Pydantic shape."""
    return ChatOllama(
        model=model_id,
        num_ctx=num_ctx,
        num_predict=num_predict,
        reasoning=reasoning,
        temperature=0,
    ).with_structured_output(
        schema,
        method="json_schema",
    )
