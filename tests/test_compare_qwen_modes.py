from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

INDEXING_EXPERIMENTS = Path(__file__).parents[1] / "experiments" / "indexing"
sys.path.insert(0, str(INDEXING_EXPERIMENTS))

comparison = importlib.import_module("compare_qwen_modes")

from src.retrieval.methods import list_retrievers  # noqa: E402


def test_every_benchmark_group_names_a_real_method():
    """A group naming a deleted method fails the run only after the model loads."""
    catalog = set(list_retrievers())
    grouped = {*comparison.DEFAULT_METHODS, *comparison.PHASE2_METHODS}

    assert grouped <= catalog, sorted(grouped - catalog)


def test_judge_scores_share_column_but_use_method_specific_hits():
    run = comparison.EvalRun(
        questions=[],
        relevance=[],
        methods=["rag-a", "rag-b"],
        warmup_count=0,
        rankings={},
        timings={},
        score_names=[],
        scores={"rag-a": {}, "rag-b": {}},
    )

    judged = comparison.add_judge_adjusted_scores(
        run,
        summaries={
            "rag-a": {"questions": 2, "strict_hits": 1, "equivalent_misses": 0},
            "rag-b": {"questions": 2, "strict_hits": 1, "equivalent_misses": 1},
        },
        cutoff=5,
    )

    assert judged.score_names == ["judge_hit@5"]
    assert judged.scores["rag-a"] == {"judge_hit@5": 0.5}
    assert judged.scores["rag-b"] == {"judge_hit@5": 1.0}


def test_report_uses_one_shared_metric_column_family():
    run = comparison.EvalRun(
        questions=[],
        relevance=[],
        methods=["rag-a", "rag-b"],
        warmup_count=0,
        rankings={},
        timings={
            method: {
                "seconds": 0.0,
                "ms_per_query": 0.0,
                "queries_per_query": 1.0,
                "total_queries": 0.0,
            }
            for method in ("rag-a", "rag-b")
        },
        score_names=["hit@1", "hit@5", "recall@5", "mrr@5", "judge_hit@5"],
        scores={
            "rag-a": {
                "hit@1": 0.0,
                "hit@5": 0.5,
                "recall@5": 0.5,
                "mrr@5": 0.25,
                "judge_hit@5": 0.5,
            },
            "rag-b": {
                "hit@1": 1.0,
                "hit@5": 1.0,
                "recall@5": 1.0,
                "mrr@5": 1.0,
                "judge_hit@5": 1.0,
            },
        },
    )

    report = comparison.format_eval_report(run, include_categories=False)

    assert report.count("hit@1") == 1
    assert report.count("judge_hit@5") == 1


def test_checkpoint_is_extended_with_a_newly_added_method():
    previous_signature = {
        "version": 3,
        "category": None,
        "limit": None,
        "warmup": 5,
        "warmup_ids": ["warmup"],
        "timed_ids": ["q1"],
        "result_limit": 15,
        "scoring_unit": "chunk",
    }
    state = {
        "signature": previous_signature,
        "methods": {"rag-a": comparison._empty_method_state()},
    }
    signature = {**previous_signature, "method_names": ["rag-a", "rag-b"]}

    assert comparison._extend_compatible_state(state, signature, ["rag-a", "rag-b"])
    assert state["signature"] == signature
    assert set(state["methods"]) == {"rag-a", "rag-b"}


def test_checkpoint_from_a_different_scoring_unit_is_rejected():
    """Legacy checkpoints written under a non-chunk scoring unit must not resume."""
    previous_signature = {
        "version": 3,
        "category": None,
        "limit": None,
        "warmup": 5,
        "warmup_ids": ["warmup"],
        "timed_ids": ["q1"],
        "result_limit": 15,
        "scoring_unit": "okf_concept",
    }
    state = {
        "signature": previous_signature,
        "methods": {"rag-a": comparison._empty_method_state()},
    }
    signature = {**previous_signature, "scoring_unit": "chunk"}

    assert not comparison._extend_compatible_state(state, signature, ["rag-a"])


def test_catch_up_plan_only_advances_new_method(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "methods": {
                    "rag-a": {"next_index": 200},
                    "rag-b": {"next_index": 199},
                }
            }
        )
    )

    assert comparison._catch_up_plan(
        checkpoint,
        ["rag-a", "rag-b", "rag-c"],
    ) == {
        "rag-a": 200,
        "rag-b": 199,
        "rag-c": 199,
    }
