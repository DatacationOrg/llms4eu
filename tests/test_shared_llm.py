"""Retry, circuit-breaker and keep-alive behaviour of the structured judge wrapper."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.shared import llm
from src.shared.llm import JudgeCircuitOpen, LocalOllamaStructuredLlm


class Verdict(BaseModel):
    ok: bool


class _FakeModel:
    def __init__(self, outcomes: list, calls: list[dict]) -> None:
        self._outcomes = outcomes
        self._calls = calls

    def invoke(self, prompt):
        outcome = self._outcomes.pop(0)
        self._calls[-1]["prompt"] = prompt
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch(monkeypatch, outcomes: list) -> list[dict]:
    calls: list[dict] = []

    def fake_factory(model_id, schema, **kwargs):
        calls.append(dict(kwargs))
        return _FakeModel(outcomes, calls)

    monkeypatch.setattr(llm, "structured_local_model", fake_factory)
    return calls


def test_first_attempt_is_deterministic_and_retries_sample(monkeypatch):
    calls = _patch(monkeypatch, [ValueError("bad json"), None, Verdict(ok=True)])
    judge = LocalOllamaStructuredLlm("m", retry_temperature=0.4, keep_alive="24h")

    result = judge.structured_output("q", Verdict, retries=3)

    assert result == Verdict(ok=True)
    assert [c["temperature"] for c in calls] == [0, 0.4, 0.4]
    assert all(c["keep_alive"] == "24h" for c in calls)
    # Retries carry the correction; the first attempt is the bare prompt.
    assert calls[0]["prompt"] == "q"
    assert "bad json" in calls[1]["prompt"]
    assert "no structured output" in calls[2]["prompt"]
    assert judge.failure_stats == (1, 0)


def test_exhausted_retries_raise_runtime_error(monkeypatch):
    _patch(monkeypatch, [ValueError("x")] * 2)
    judge = LocalOllamaStructuredLlm("m")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        judge.structured_output("q", Verdict, retries=2)
    assert judge.failure_stats == (1, 1)


def test_breaker_trips_only_after_min_calls(monkeypatch):
    # 20 calls: 15 failures, 5 successes -> 75% once the window is full.
    outcomes: list = []
    for i in range(20):
        outcomes.append(ValueError("x") if i % 4 else Verdict(ok=True))
    _patch(monkeypatch, outcomes)
    judge = LocalOllamaStructuredLlm("m", max_failure_rate=0.5, breaker_min_calls=20)

    for i in range(19):
        if i % 4:
            with pytest.raises(RuntimeError):
                judge.structured_output("q", Verdict, retries=1)
        else:
            judge.structured_output("q", Verdict, retries=1)
    # The 20th call is a failure and pushes the rate over the threshold.
    with pytest.raises(JudgeCircuitOpen, match="failed 15 of 20"):
        judge.structured_output("q", Verdict, retries=1)


def test_breaker_disabled_by_default(monkeypatch):
    _patch(monkeypatch, [ValueError("x")] * 30)
    judge = LocalOllamaStructuredLlm("m", breaker_min_calls=1)
    for _ in range(30):
        with pytest.raises(RuntimeError):
            judge.structured_output("q", Verdict, retries=1)


def test_circuit_open_is_not_a_runtime_error():
    # The retrievers catch RuntimeError and fall back; the breaker must escape.
    assert not issubclass(JudgeCircuitOpen, RuntimeError)


# --- Azure AI Foundry judge -------------------------------------------------

from src.shared.llm import AzureFoundryStructuredLlm  # noqa: E402


def _azure(monkeypatch, replies: list) -> tuple[AzureFoundryStructuredLlm, list[dict]]:
    calls: list[dict] = []

    def fake_request(self, prompt, output_schema, temperature):
        calls.append({"prompt": prompt, "temperature": temperature})
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(AzureFoundryStructuredLlm, "_request", fake_request)
    judge = AzureFoundryStructuredLlm(
        endpoint="https://x.test/openai/v1", api_key="k", model="m"
    )
    return judge, calls


def test_azure_parses_json_even_when_wrapped_in_prose(monkeypatch):
    judge, calls = _azure(
        monkeypatch, ['Sure, here it is:\n```json\n{"ok": true}\n```']
    )
    assert judge.structured_output("q", Verdict) == Verdict(ok=True)
    assert calls[0]["temperature"] == 0
    assert judge.failure_stats == (1, 0)


def test_azure_retries_with_correction_and_sampling(monkeypatch):
    judge, calls = _azure(
        monkeypatch, ["no json here", '{"ok": "not-a-bool"}', '{"ok": false}']
    )
    assert judge.structured_output("q", Verdict, retries=3) == Verdict(ok=False)
    assert [c["temperature"] for c in calls] == [0, 0.3, 0.3]
    assert calls[0]["prompt"] == "q"
    assert "did not contain a JSON object" in calls[1]["prompt"]
    assert "ok" in calls[2]["prompt"]  # the required-field list from the schema


def test_azure_exhausted_retries_raise_runtime_error_and_trip_breaker(monkeypatch):
    judge, _ = _azure(monkeypatch, ["nope"] * 4)
    judge = AzureFoundryStructuredLlm(
        endpoint="https://x.test",
        api_key="k",
        model="m",
        max_failure_rate=0.0,
        breaker_min_calls=2,
    )
    monkeypatch.setattr(
        AzureFoundryStructuredLlm, "_request", lambda self, p, s, t: "nope"
    )
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        judge.structured_output("q", Verdict, retries=1)
    with pytest.raises(JudgeCircuitOpen, match="failed 2 of 2"):
        judge.structured_output("q", Verdict, retries=1)


def test_azure_from_env_names_the_missing_variables(monkeypatch):
    monkeypatch.setattr(llm, "load_local_env", lambda: None)  # do not read .env
    for name in ("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_AI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(
        RuntimeError, match="AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, AZURE_AI_MODEL"
    ):
        AzureFoundryStructuredLlm.from_env()


def test_azure_repr_hides_the_key():
    judge = AzureFoundryStructuredLlm(
        endpoint="https://x.test", api_key="secret", model="m"
    )
    assert "secret" not in repr(judge)


# --- default_judge provider switch -------------------------------------------

from src.retrieval.retrievers.agentic import default_judge  # noqa: E402


def test_default_judge_builds_azure_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_AI_ENDPOINT", "https://x.test/openai/v1/")
    monkeypatch.setenv("AZURE_AI_API_KEY", "k")
    monkeypatch.setenv("AZURE_AI_MODEL", "DeepSeek-V4-Pro")
    judge = default_judge(
        {
            "agentic_judge_provider": "azure",
            "agentic_judge_num_predict": 4096,
            "agentic_judge_max_failure_rate": 0.2,
        }
    )
    assert isinstance(judge, AzureFoundryStructuredLlm)
    assert judge.model_id == "azure:DeepSeek-V4-Pro"
    assert judge.max_tokens == 4096
    assert judge.max_failure_rate == 0.2


def test_default_judge_builds_ollama_when_asked():
    judge = default_judge(
        {"agentic_judge_provider": "ollama", "agentic_judge_model": "gpt-oss:20b"}
    )
    assert isinstance(judge, LocalOllamaStructuredLlm)
    assert judge.model_id == "gpt-oss:20b"


def test_default_judge_rejects_unknown_provider():
    with pytest.raises(ValueError, match="agentic_judge_provider"):
        default_judge({"agentic_judge_provider": "bedrock"})
