from __future__ import annotations

import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from langchain_ollama import ChatOllama
from pydantic import BaseModel

__all__ = [
    "AzureFoundryStructuredLlm",
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
        )
        return model.with_retry(stop_after_attempt=retries).invoke(_messages(prompt))


@dataclass(frozen=True)
class AzureFoundryStructuredLlm:
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: int = 90
    temperature: float = 0
    max_tokens: int = 1024

    @classmethod
    def from_env(cls, **overrides) -> AzureFoundryStructuredLlm:
        return cls(
            endpoint=os.environ["AZURE_AI_ENDPOINT"],
            api_key=os.environ["AZURE_AI_API_KEY"],
            model=os.environ["AZURE_AI_MODEL"],
            **overrides,
        )

    def structured_output(
        self,
        prompt: StructuredPrompt,
        output_schema: type[T],
        *,
        retries: int = 3,
    ) -> T:
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                content = self._request(prompt, output_schema)
                return output_schema.model_validate_json(_json_object(content))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"structured Azure Foundry call failed after {retries} attempts: {detail}"
        ) from last_error

    def _request(self, prompt: StructuredPrompt, output_schema: type[BaseModel]) -> str:
        payload = {
            "model": self.model,
            "messages": _azure_messages(prompt, output_schema),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.endpoint.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


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


def _messages(prompt: StructuredPrompt) -> StructuredPrompt:
    if isinstance(prompt, str):
        return prompt
    return list(prompt)


def _azure_messages(
    prompt: StructuredPrompt,
    output_schema: type[BaseModel],
) -> list[dict[str, str]]:
    schema_instruction = (
        "Return only a JSON object matching this schema: "
        f"{output_schema.model_json_schema()}"
    )
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": schema_instruction},
            {"role": "user", "content": prompt},
        ]
    messages = [
        {"role": _azure_role(role), "content": content} for role, content in prompt
    ]
    return [{"role": "system", "content": schema_instruction}, *messages]


def _azure_role(role: str) -> str:
    return "user" if role == "human" else role


def _json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("response did not contain a JSON object")
    return match.group(0)
