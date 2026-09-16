from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_agentic_diagnostics(
    *,
    questions: list[dict[str, Any]],
    relevance: list[dict[str, Any]],
    rankings: dict[str, dict[str, list[str]]],
    method_states: dict[str, dict[str, Any]],
    method_names: list[str],
    cutoff: int,
) -> dict[str, Any]:
    """Compare each selected agent with its matching non-agentic baseline."""
    relevant_by_question: dict[str, set[str]] = defaultdict(set)
    for row in relevance:
        relevant_by_question[str(row["question_id"])].add(str(row["chunk_id"]))
    question_by_id = {str(row["id"]): row for row in questions}

    summaries: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    selected = set(method_names)
    for agent_name in method_names:
        baseline_name = baseline_for_agent(agent_name)
        if baseline_name is None or baseline_name not in selected:
            continue

        agent_rankings = rankings.get(agent_name, {})
        baseline_rankings = rankings.get(baseline_name, {})
        observations = _observations(method_states.get(agent_name, {}))
        rows = []
        for question_id in sorted(set(agent_rankings) & set(baseline_rankings)):
            relevant = relevant_by_question.get(str(question_id), set())
            if not relevant:
                continue
            observation = observations.get(str(question_id), {})
            actions = observation.get("actions", [])
            row = _diagnostic_row(
                question=question_by_id.get(str(question_id), {"id": question_id}),
                relevant=relevant,
                baseline_ranking=baseline_rankings[question_id],
                agent_ranking=agent_rankings[question_id],
                actions=actions,
                elapsed_seconds=observation.get("elapsed_seconds"),
                query_count=observation.get("query_count"),
                cutoff=cutoff,
            )
            rows.append(row)

        details[agent_name] = rows
        summaries[agent_name] = _summarize(rows, baseline_name, cutoff)

    return {"cutoff": cutoff, "summaries": summaries, "details": details}


# Reasoning rungs and named judges are agent-only axes: there is no
# `*_rerank_high` or `*_rerank_gptoss`, because the reranker runs no LLM. Those
# suffixes therefore have to come off before the baseline name is derived, or
# every rung would pair with a method that does not exist and the diagnostics
# would report no pairs at all rather than an error.
REASONING_SUFFIXES = ("_high", "_medium", "_low")


def judge_suffixes() -> tuple[str, ...]:
    """`_{name}` for every judge in `agentic_judges` (src/retrieval/config.yaml)."""
    from src.shared.env import ROOT, load_yaml

    config = load_yaml(ROOT / "src" / "retrieval" / "config.yaml")
    return tuple(f"_{name}" for name in (config.get("agentic_judges") or {}))


def strip_reasoning_suffix(method_name: str) -> tuple[str, str | None]:
    """Split a method name into its base name and its rung or judge, if any."""
    for suffix in (*REASONING_SUFFIXES, *judge_suffixes()):
        if suffix in method_name:
            head, _, tail = method_name.partition(suffix)
            return head + tail, suffix.lstrip("_")
    return method_name, None


def baseline_for_agent(method_name: str) -> str | None:
    if "_agentic" not in method_name:
        return None
    name, _ = strip_reasoning_suffix(method_name)
    # The tool-using agent pairs with the same reranked baseline as the plain
    # agent, so `_tools` must come off before the `_agentic` swap.
    base = name.replace("_agentic_tools", "_agentic", 1)
    return base.replace("_agentic", "_rerank", 1)


def format_agentic_diagnostics(diagnostics: dict[str, Any]) -> str:
    summaries = diagnostics.get("summaries", {})
    if not summaries:
        return ""
    cutoff = diagnostics["cutoff"]
    headers = [
        "agent",
        "baseline",
        "queries",
        "retried",
        "rewritten",
        "expanded",
        "retry precision",
        "retry recall",
        f"recovered@{cutoff}",
        f"lost@{cutoff}",
        "retry ms",
        "no-retry ms",
    ]
    rows = []
    for agent_name, summary in summaries.items():
        rows.append(
            [
                agent_name,
                summary["baseline"],
                str(summary["questions"]),
                _rate_count(summary["retried"], summary["questions"]),
                str(summary["rewritten"]),
                str(summary["expanded"]),
                _rate(summary["retry_precision"]),
                _rate(summary["retry_recall"]),
                str(summary["recovered"]),
                str(summary["lost"]),
                _number(summary["retry_ms"]),
                _number(summary["no_retry_ms"]),
            ]
        )
    sections = [
        f"Agentic diagnostics (paired at hit@{cutoff})",
        "",
        _markdown_table(headers, rows),
        "",
        "Retry precision is the fraction of retried queries whose first relevant rank improved. "
        "Retry recall is the fraction of baseline misses that were retried.",
    ]
    tools = _format_tool_diagnostics(diagnostics)
    if tools:
        sections.extend(["", tools])
    return "\n".join(sections)


def _format_tool_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Page-tool usage per agent, omitted entirely when no agent has tools."""
    summaries = diagnostics.get("summaries", {})
    cutoff = diagnostics["cutoff"]
    active = {
        name: summary
        for name, summary in summaries.items()
        if summary.get("tool_calls")
    }
    if not active:
        return ""

    headers = [
        "agent",
        "questions",
        "tool questions",
        "tool calls",
        "list_sections",
        "search_in_page",
        f"recovered@{cutoff}",
        f"lost@{cutoff}",
        "tool precision",
    ]
    rows = [
        [
            name,
            str(summary["questions"]),
            _rate_count(summary["tool_questions"], summary["questions"]),
            str(summary["tool_calls"]),
            str(summary["list_sections_calls"]),
            str(summary["search_in_page_calls"]),
            str(summary["tool_recovered"]),
            str(summary["tool_lost"]),
            _rate(summary["tool_precision"]),
        ]
        for name, summary in active.items()
    ]
    return "\n".join(
        [
            "Page tool usage",
            "",
            _markdown_table(headers, rows),
            "",
            "Tool questions are questions where the agent called a page tool. "
            f"Tool precision is recovered@{cutoff} over those questions; per-question "
            "action sequences and search terms are in the diagnostics JSON.",
        ]
    )


def _diagnostic_row(
    *,
    question: dict[str, Any],
    relevant: set[str],
    baseline_ranking: list[str],
    agent_ranking: list[str],
    actions: list[dict[str, Any]],
    elapsed_seconds: float | None,
    query_count: float | None,
    cutoff: int,
) -> dict[str, Any]:
    baseline_rank = _first_rank(baseline_ranking, relevant)
    agent_rank = _first_rank(agent_ranking, relevant)
    fallback_rank = max(len(baseline_ranking), len(agent_ranking), cutoff) + 1
    baseline_value = baseline_rank or fallback_rank
    agent_value = agent_rank or fallback_rank
    retried = len(actions) > 1
    rewritten, expanded = _action_types(actions)
    scores = (
        [float(chunk["score"]) for chunk in actions[0].get("chunks", [])]
        if actions
        else []
    )
    final_verdict = actions[-1].get("verdict", {}) if actions else {}
    baseline_hit = baseline_rank is not None and baseline_rank <= cutoff
    agent_hit = agent_rank is not None and agent_rank <= cutoff
    return {
        "question_id": str(question["id"]),
        "category": question.get("question_type"),
        "baseline_hit": baseline_hit,
        "agent_hit": agent_hit,
        "baseline_rank": baseline_rank,
        "agent_rank": agent_rank,
        "rank_delta": baseline_value - agent_value,
        "retried": retried,
        "rewritten": rewritten,
        "expanded": expanded,
        "recovered": not baseline_hit and agent_hit,
        "lost": baseline_hit and not agent_hit,
        "query_count": query_count,
        "elapsed_ms": elapsed_seconds * 1000 if elapsed_seconds is not None else None,
        "initial_top_score": scores[0] if scores else None,
        "initial_top1_top2_margin": scores[0] - scores[1] if len(scores) > 1 else None,
        "initial_score_spread": scores[0] - scores[-1] if len(scores) > 1 else None,
        "final_sufficient": final_verdict.get("sufficient"),
        "final_reason": final_verdict.get("reason"),
        **_tool_usage(actions),
        "retrieval_ms": sum(
            float(action.get("retrieval_ms", 0.0)) for action in actions
        ),
        "judge_ms": sum(float(action.get("judge_ms", 0.0)) for action in actions),
        "top_up_ms": sum(float(action.get("top_up_ms", 0.0)) for action in actions),
        "actions": actions,
    }


def _action_types(actions: list[dict[str, Any]]) -> tuple[bool, bool]:
    rewritten = False
    expanded = False
    for previous, current in zip(actions, actions[1:]):
        rewritten |= current.get("judge_query") != previous.get("judge_query")
        expanded |= len(current.get("chunks", [])) > len(previous.get("chunks", []))
    return rewritten, expanded


TOOL_NAMES = ("list_sections", "search_in_page")


def _tool_usage(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-question tool trace, empty for agents without tools."""
    steps = [action.get("verdict", {}) for action in actions]
    sequence = [str(step.get("action")) for step in steps if step.get("action")]
    calls = [step for step in steps if step.get("action") in TOOL_NAMES]
    return {
        "action_sequence": sequence,
        "tool_calls": len(calls),
        "tools_used": sorted({str(step["action"]) for step in calls}),
        "tool_details": [
            {
                "tool": step.get("action"),
                "page_id": step.get("page_id"),
                "term": step.get("term"),
                "reason": step.get("reason"),
            }
            for step in calls
        ],
    }


def _summarize_tools(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Did page tools get used, and did using them pay off?"""
    used = [row for row in rows if row.get("tool_calls")]
    return {
        "tool_calls": sum(row.get("tool_calls", 0) for row in rows),
        "tool_questions": len(used),
        "list_sections_calls": sum(
            row.get("action_sequence", []).count("list_sections") for row in rows
        ),
        "search_in_page_calls": sum(
            row.get("action_sequence", []).count("search_in_page") for row in rows
        ),
        # Recoveries on questions where a tool ran: the payoff signal that says
        # whether the extra latency bought anything.
        "tool_recovered": sum(row["recovered"] for row in used),
        "tool_lost": sum(row["lost"] for row in used),
        "tool_precision": _divide(sum(row["recovered"] for row in used), len(used)),
    }


def _summarize(
    rows: list[dict[str, Any]],
    baseline: str,
    cutoff: int,
    *,
    include_categories: bool = True,
) -> dict[str, Any]:
    retried = [row for row in rows if row["retried"]]
    baseline_misses = [row for row in rows if not row["baseline_hit"]]
    improved_retries = [row for row in retried if row["rank_delta"] > 0]
    retried_baseline_misses = [row for row in baseline_misses if row["retried"]]
    summary = {
        "baseline": baseline,
        "cutoff": cutoff,
        "questions": len(rows),
        "retried": len(retried),
        "rewritten": sum(row["rewritten"] for row in rows),
        "expanded": sum(row["expanded"] for row in rows),
        "retry_precision": _divide(len(improved_retries), len(retried)),
        "retry_recall": _divide(len(retried_baseline_misses), len(baseline_misses)),
        "recovered": sum(row["recovered"] for row in rows),
        "lost": sum(row["lost"] for row in rows),
        "rank_improved": sum(row["rank_delta"] > 0 for row in rows),
        "rank_worsened": sum(row["rank_delta"] < 0 for row in rows),
        "retry_ms": _mean(row["elapsed_ms"] for row in retried),
        "no_retry_ms": _mean(row["elapsed_ms"] for row in rows if not row["retried"]),
        "retrieval_ms": _mean(row["retrieval_ms"] for row in rows),
        "judge_ms": _mean(row["judge_ms"] for row in rows),
        "top_up_ms": _mean(row["top_up_ms"] for row in rows),
        "sufficient_rate": _mean(
            1.0 if row["final_sufficient"] else 0.0
            for row in rows
            if row["final_sufficient"] is not None
        ),
        **_summarize_tools(rows),
    }
    if include_categories:
        categories = sorted({row["category"] for row in rows if row["category"]})
        summary["by_category"] = {
            category: _summarize(
                [row for row in rows if row["category"] == category],
                baseline,
                cutoff,
                include_categories=False,
            )
            for category in categories
        }
    return summary


def _observations(method_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observations = method_state.get("observations", {})
    if observations:
        return {str(key): value for key, value in observations.items()}

    rankings = method_state.get("rankings", {})
    groups: list[list[dict[str, Any]]] = []
    for action in method_state.get("action_log", []):
        if action.get("attempt") == 1 or not groups:
            groups.append([])
        groups[-1].append(action)
    return {
        str(question_id): {"actions": actions}
        for question_id, actions in zip(rankings, groups, strict=False)
    }


def _first_rank(ranking: list[str], relevant: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant:
            return index
    return None


def _divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _rate_count(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else "0 (-)"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("-" * (len(header) + 2) for header in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
