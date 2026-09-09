"""What `run_eval` exposes about an agent's behaviour, and category scoring.

`retriever_action_log` and `category_hit_scores` exist so a chunk-variant grid can
report agentic diagnostics and the category split without re-implementing either.
Both were previously reachable only from inside the single-run report, which is
how the variant sweep ended up without those sections at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.evaluate import (
    EvalRun,
    category_hit_scores,
    category_question_types,
    retriever_action_log,
)


@dataclass
class _BatchStats:
    action_log: list[dict] = field(default_factory=list)


class _Agent:
    """An agent that keeps its log on `batch_stats`, as AgenticRetriever does."""

    def __init__(self, entries):
        self.batch_stats = _BatchStats(action_log=list(entries))


class _FlatAgent:
    """An agent that keeps `action_log` directly, without an AgenticBatchStats."""

    def __init__(self, entries):
        self.action_log = list(entries)


class _Plain:
    name = "qwen"


def test_an_agent_keeping_its_log_on_batch_stats_is_read():
    log = retriever_action_log(_Agent([{"action": "reformulate"}]))
    assert log == [{"action": "reformulate"}]


def test_an_agent_keeping_the_log_directly_is_also_read():
    # Duck-typed on purpose: not every agent is an AgenticRetriever subclass.
    assert retriever_action_log(_FlatAgent([{"action": "search_in_page"}])) == [
        {"action": "search_in_page"}
    ]


def test_a_non_agentic_retriever_reports_none_rather_than_an_empty_log():
    # None and [] mean different things downstream: "keeps no log" versus "ran and
    # never acted". A vector retriever must not look like an agent that idled.
    assert retriever_action_log(_Plain()) is None


def test_the_log_is_copied_so_a_later_batch_cannot_rewrite_a_finished_cell():
    # The retriever clears and reuses its log per batch, so a live reference would
    # leave a completed variant's diagnostics pointing at the next variant's run.
    agent = _Agent([{"action": "expand"}])
    captured = retriever_action_log(agent)
    agent.batch_stats.action_log.clear()
    agent.batch_stats.action_log.append({"action": "sufficient"})
    assert captured == [{"action": "expand"}]


def _run() -> EvalRun:
    questions = [
        {"id": "q1", "question_type": "crosslingual"},
        {"id": "q2", "question_type": "crosslingual"},
        {"id": "q3", "question_type": "vague_short"},
    ]
    relevance = [
        {"question_id": "q1", "chunk_id": "c1"},
        {"question_id": "q2", "chunk_id": "c2"},
        {"question_id": "q3", "chunk_id": "c3"},
    ]
    rankings = {
        "qwen": {"q1": ["c1"], "q2": ["zz"], "q3": ["c3"]},
    }
    return EvalRun(
        questions=questions,
        relevance=relevance,
        methods=["qwen"],
        warmup_count=0,
        rankings=rankings,
        timings={},
        score_names=[],
        scores={"qwen": {}},
    )


def test_category_types_come_back_in_a_stable_order():
    assert category_question_types(_run()) == ["crosslingual", "vague_short"]


def test_category_scores_are_computed_per_type_not_over_the_whole_run():
    # One of two crosslingual questions hit, and the single vague_short one did.
    # Averaging over the run would give 2/3 for both and hide the difference the
    # category split exists to show.
    scores = category_hit_scores(_run())
    assert scores["qwen"]["crosslingual"] == 0.5
    assert scores["qwen"]["vague_short"] == 1.0
