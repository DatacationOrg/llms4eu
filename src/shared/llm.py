from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol, TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel

__all__ = [
    "LocalOllamaStructuredLlm",
    "StructuredLlm",
    "StructuredPrompt",
    "run_structured_outputs",
    "structured_local_model",
]

T = TypeVar("T", bound=BaseModel)
Message = tuple[str, str]
StructuredPrompt = str | Sequence[Message]


class StructuredLlm(Protocol):
    def structured_output(
        self,
        prompt: StructuredPrompt,
        output_schema: type[T],
        *,
        retries: int = 3,
    ) -> T: ...


@dataclass(frozen=True)
class LocalOllamaStructuredLlm:
    model_id: str
    reasoning: bool | str | None = True
    num_ctx: int | None = None
    num_predict: int | None = None
    method: str = "json_schema"

    def structured_output(
        self,
        prompt: StructuredPrompt,
        output_schema: type[T],
        *,
        retries: int = 3,
    ) -> T:
        model = structured_local_model(
            self.model_id,
            output_schema,
            reasoning=self.reasoning,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            method=self.method,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            # Temperature is 0 for reproducibility, so an identical prompt
            # reproduces an identical failure. Each retry must name what went
            # wrong, or the extra attempts are wasted calls.
            retry_prompt = (
                prompt
                if attempt == 0
                else _with_correction(prompt, last_error, output_schema)
            )
            try:
                result = model.invoke(_messages(retry_prompt))
            except Exception as exc:  # transport, parse, and validation failures
                last_error = exc
                continue
            # function_calling yields None when the model answers without
            # calling the tool; that is a failed attempt, not a valid result.
            if result is not None:
                return result
            last_error = ValueError("model returned no structured output")
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"structured call to {self.model_id} failed after "
            f"{max(1, retries)} attempts: {detail}"
        ) from last_error


def run_structured_outputs(
    client: StructuredLlm,
    prompts: Sequence[StructuredPrompt],
    output_schema: type[T],
    *,
    workers: int = 4,
    retries: int = 3,
) -> list[T]:
    results: list[T | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                client.structured_output,
                prompt,
                output_schema,
                retries=retries,
            ): index
            for index, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]


def structured_local_model(
    model_id: str,
    schema: type[BaseModel],
    reasoning: bool | str | None = True,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    method: str = "json_schema",
):
    """Create a local Ollama chat model that returns the requested Pydantic shape.

    Use `method="function_calling"` for models that ignore `json_schema` when
    reasoning is disabled; Gemma 4 answers such prompts in prose instead of JSON.
    """
    return ChatOllama(
        model=model_id,
        num_ctx=num_ctx,
        num_predict=num_predict,
        reasoning=reasoning,
        temperature=0,
    ).with_structured_output(schema, method=method)


def _messages(prompt: StructuredPrompt) -> StructuredPrompt:
    if isinstance(prompt, str):
        return prompt
    return list(prompt)


def _with_correction(
    prompt: StructuredPrompt,
    error: Exception | None,
    output_schema: type[BaseModel],
) -> StructuredPrompt:
    """Tell the model what it got wrong so the retry differs from the attempt."""
    required = output_schema.model_json_schema().get("required", [])
    instruction = (
        "The previous response was rejected: "
        f"{type(error).__name__}: {error}. "
        "Answer again by calling the tool with every required field present"
        + (f": {', '.join(required)}." if required else ".")
    )
    if isinstance(prompt, str):
        return f"{prompt}\n\n{instruction}"
    return [*prompt, ("human", instruction)]
