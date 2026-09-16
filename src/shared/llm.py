from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import os
import re
import threading
from typing import Protocol, TypeVar

import httpx
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.shared.env import load_local_env

__all__ = [
    "AzureFoundryStructuredLlm",
    "JudgeCircuitOpen",
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


class JudgeCircuitOpen(Exception):
    """Too many structured calls have failed for this judge to be trusted.

    Deliberately not a `RuntimeError`: the agentic retrievers catch that and
    fall back, which is right for one bad question and wrong for a judge that
    fails on every other one. A run whose judge is broken must stop, not spend
    the night scoring BM25 fallback under the agent's name (the 2026-09-01
    gemma run: 136 failures, 9.5 hours, one cell).
    """


@dataclass
class _FailureCounter:
    calls: int = 0
    failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class LocalOllamaStructuredLlm:
    model_id: str
    reasoning: bool | str | None = True
    num_ctx: int | None = None
    num_predict: int | None = None
    method: str = "json_schema"
    # The first attempt is at temperature 0 so a run is reproducible. Retries
    # sample instead: an identical prompt at temperature 0 reproduces an
    # identical failure, so a retry that only changes the wording of the
    # corrective note rarely lands.
    retry_temperature: float = 0.3
    # Ollama unloads an idle model after about five minutes. Between cells the
    # judge sits idle, and if the GPU has been taken meanwhile the reload fails
    # for the rest of that cell. This is passed on every request.
    keep_alive: int | str | None = None
    # Circuit breaker: once at least `breaker_min_calls` calls have been made,
    # a failure rate (after retries) above `max_failure_rate` raises
    # `JudgeCircuitOpen` instead of returning another fallback. `None` disables.
    max_failure_rate: float | None = None
    breaker_min_calls: int = 20
    _counter: _FailureCounter = field(
        default_factory=_FailureCounter, init=False, repr=False, compare=False
    )

    @property
    def failure_stats(self) -> tuple[int, int]:
        """(calls, failures-after-retries) seen by this judge so far."""
        return self._counter.calls, self._counter.failures

    def structured_output(
        self,
        prompt: StructuredPrompt,
        output_schema: type[T],
        *,
        retries: int = 3,
    ) -> T:
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            model = structured_local_model(
                self.model_id,
                output_schema,
                reasoning=self.reasoning,
                num_ctx=self.num_ctx,
                num_predict=self.num_predict,
                method=self.method,
                temperature=0 if attempt == 0 else self.retry_temperature,
                keep_alive=self.keep_alive,
            )
            # Each retry must also name what went wrong, or the extra attempt
            # only differs by sampling noise.
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
                self._record(failed=False)
                return result
            last_error = ValueError("model returned no structured output")
        self._record(failed=True)
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"structured call to {self.model_id} failed after "
            f"{max(1, retries)} attempts: {detail}"
        ) from last_error

    def _record(self, *, failed: bool) -> None:
        _record_outcome(
            self._counter,
            failed=failed,
            label=self.model_id,
            max_failure_rate=self.max_failure_rate,
            min_calls=self.breaker_min_calls,
        )


def _record_outcome(
    counter: _FailureCounter,
    *,
    failed: bool,
    label: str,
    max_failure_rate: float | None,
    min_calls: int,
) -> None:
    with counter.lock:
        counter.calls += 1
        counter.failures += int(failed)
        calls, failures = counter.calls, counter.failures
    if (
        failed
        and max_failure_rate is not None
        and calls >= min_calls
        and failures / calls > max_failure_rate
    ):
        raise JudgeCircuitOpen(
            f"{label} failed {failures} of {calls} structured calls "
            f"({failures / calls:.0%} > {max_failure_rate:.0%}); "
            "stopping rather than scoring fallback under this method's name"
        )


@dataclass(frozen=True)
class AzureFoundryStructuredLlm:
    """Structured output from an OpenAI-compatible Azure AI Foundry deployment.

    This was the original sufficiency judge (DeepSeek on Azure) before the local
    Ollama judges; it is back because the local models cost a shared GPU and
    weeks of reliability work. The endpoint speaks the `chat/completions`
    protocol with `response_format: json_object`; the schema is put in a system
    message, the reply is validated against the Pydantic model, and a reasoning
    model's `reasoning_content` is used when `content` holds no JSON.

    Environment: `AZURE_AI_ENDPOINT`, `AZURE_AI_API_KEY`, `AZURE_AI_MODEL`.
    """

    endpoint: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: int = 90
    temperature: float = 0
    max_tokens: int = 1024
    retry_temperature: float = 0.3
    max_failure_rate: float | None = None
    breaker_min_calls: int = 20
    _counter: _FailureCounter = field(
        default_factory=_FailureCounter, init=False, repr=False, compare=False
    )

    @classmethod
    def from_env(cls, **overrides) -> AzureFoundryStructuredLlm:
        # Nothing on the benchmark path loads .env otherwise; without this the
        # judge only works in shells that happen to have the variables exported.
        load_local_env()
        missing = [
            name
            for name in ("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_AI_MODEL")
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                "Azure judge selected but the environment lacks "
                + ", ".join(missing)
                + " (set them in .env)"
            )
        return cls(
            endpoint=os.environ["AZURE_AI_ENDPOINT"],
            api_key=os.environ["AZURE_AI_API_KEY"],
            model=os.environ["AZURE_AI_MODEL"],
            **overrides,
        )

    @property
    def model_id(self) -> str:
        return f"azure:{self.model}"

    @property
    def failure_stats(self) -> tuple[int, int]:
        return self._counter.calls, self._counter.failures

    def structured_output(
        self,
        prompt: StructuredPrompt,
        output_schema: type[T],
        *,
        retries: int = 3,
    ) -> T:
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            retry_prompt = (
                prompt
                if attempt == 0
                else _with_correction(prompt, last_error, output_schema)
            )
            temperature = self.temperature if attempt == 0 else self.retry_temperature
            try:
                content = self._request(retry_prompt, output_schema, temperature)
                result = output_schema.model_validate_json(_json_object(content))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                # ValueError covers pydantic's ValidationError and _json_object.
                last_error = exc
                continue
            self._record(failed=False)
            return result
        self._record(failed=True)
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown error"
        )
        raise RuntimeError(
            f"structured call to {self.model_id} failed after "
            f"{max(1, retries)} attempts: {detail}"
        ) from last_error

    def _request(
        self,
        prompt: StructuredPrompt,
        output_schema: type[BaseModel],
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": _azure_messages(prompt, output_schema),
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()[:2000]
            raise httpx.HTTPStatusError(
                f"{exc}; response body: {detail or '(empty)'}",
                request=exc.request,
                response=exc.response,
            ) from exc
        message = response.json()["choices"][0]["message"]
        content = message.get("content") or ""
        if "{" not in content:
            content = message.get("reasoning_content") or content
        return content

    def _record(self, *, failed: bool) -> None:
        _record_outcome(
            self._counter,
            failed=failed,
            label=self.model_id,
            max_failure_rate=self.max_failure_rate,
            min_calls=self.breaker_min_calls,
        )


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
    temperature: float = 0,
    keep_alive: int | str | None = None,
    include_raw: bool = False,
):
    """Create a local Ollama chat model that returns the requested Pydantic shape.

    The working `method` is a property of the (model, schema) pair, not of the
    model: measure it with `experiments/indexing/probe_structured_output.py`
    before changing either. A wrong pairing does not raise; it returns nothing
    and the caller silently falls back.

    `include_raw=True` returns `{"raw", "parsed", "parsing_error"}` instead of
    the parsed object, so a probe can read token counts and the stop reason.
    """
    return ChatOllama(
        model=model_id,
        num_ctx=num_ctx,
        num_predict=num_predict,
        reasoning=reasoning,
        temperature=temperature,
        keep_alive=keep_alive,
    ).with_structured_output(schema, method=method, include_raw=include_raw)


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


def _azure_messages(
    prompt: StructuredPrompt,
    output_schema: type[BaseModel],
) -> list[dict[str, str]]:
    schema_instruction = (
        "Return only a JSON object matching this schema, with no commentary or "
        f"Markdown fences: {output_schema.model_json_schema()}"
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
