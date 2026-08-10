from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

INDEXING_EXPERIMENTS = Path(__file__).parents[1] / "experiments" / "indexing"
sys.path.insert(0, str(INDEXING_EXPERIMENTS))

comparison = importlib.import_module("compare_qwen_modes")
equivalence = importlib.import_module("judge_retrieval_equivalence")


def test_default_benchmark_methods_are_all_local():
    """OKF and the Azure-embedded methods are no longer part of any bench group."""
    grouped = set(comparison.DEFAULT_METHODS)
    for group in comparison.PHASE2_METHODS.values():
        grouped.update(group)

    assert "okf" not in grouped
    assert not any("embed_v4" in name for name in grouped)
    assert not any("cohere" in name for name in grouped)


def test_okf_stays_reachable_as_an_explicit_opt_in():
    assert comparison._resolve_methods("okf-only") == ["okf"]


def test_tool_agent_is_paired_with_the_plain_agent_by_default():
    assert "qwen_hybrid_agentic" in comparison.DEFAULT_METHODS
    assert "qwen_hybrid_agentic_tools" in comparison.DEFAULT_METHODS


def test_action_log_is_read_from_any_agent_with_batch_stats():
    """The tool agent is not an AgenticRetriever subclass; duck-typing must find it."""

    class Stats:
        action_log = [{"attempt": 1}]

    class ToolAgent:
        name = "qwen_hybrid_agentic_tools"
        batch_stats = Stats()

    class Plain:
        name = "sparse_rerank"

    assert comparison._retriever_action_log(ToolAgent()) == [{"attempt": 1}]
    assert comparison._retriever_action_log(Plain()) is None


def test_mixed_scoring_keeps_rag_chunks_and_scores_okf_concepts():
    relevance = [
        {"question_id": "q1", "chunk_id": "gold-chunk"},
        {"question_id": "q1", "chunk_id": "sibling-chunk"},
    ]
    rankings = {
        "rag": {"q1": ["sibling-chunk", "other-chunk", "gold-chunk"]},
        "okf": {"q1": ["destinations/other.md", "destinations/castle.md"]},
    }

    score_names, scores = comparison._score_method_rankings(
        relevance=relevance,
        rankings=rankings,
        method_names=["rag", "okf"],
        chunk_pages={
            "gold-chunk": "gold-page",
            "sibling-chunk": "sibling-page",
            "other-chunk": "other-page",
        },
        page_concepts={
            "gold-page": ["destinations/castle.md"],
            "sibling-page": ["destinations/castle.md"],
            "other-page": ["destinations/other.md"],
        },
    )

    assert score_names == [
        "hit@1",
        "hit@5",
        "hit@10",
        "recall@10",
        "mrr@10",
        "ndcg@10",
    ]
    assert scores["rag"]["hit@1"] == 1.0
    assert scores["okf"]["hit@1"] == 0.0
    assert scores["okf"]["hit@5"] == 1.0


def test_rag_sibling_page_does_not_receive_okf_concept_credit():
    _, scores = comparison._score_method_rankings(
        relevance=[{"question_id": "q1", "chunk_id": "gold-chunk"}],
        rankings={
            "rag": {"q1": ["sibling-chunk"]},
            "okf": {"q1": ["destinations/castle.md"]},
        },
        method_names=["rag", "okf"],
        chunk_pages={
            "gold-chunk": "gold-page",
            "sibling-chunk": "sibling-page",
        },
        page_concepts={
            "gold-page": ["destinations/castle.md"],
            "sibling-page": ["destinations/castle.md"],
        },
    )

    assert scores["rag"]["hit@1"] == 0.0
    assert scores["okf"]["hit@1"] == 1.0


def test_mixed_judge_scores_share_column_but_use_method_specific_hits():
    run = comparison.EvalRun(
        questions=[],
        relevance=[],
        methods=["rag", "okf"],
        warmup_count=0,
        rankings={},
        timings={},
        score_names=[],
        scores={"rag": {}, "okf": {}},
    )

    judged = comparison._add_method_judge_scores(
        run,
        summaries={
            "rag": {"questions": 2, "strict_hits": 1, "equivalent_misses": 0},
            "okf": {"questions": 2, "strict_hits": 1, "equivalent_misses": 1},
        },
        cutoff=5,
    )

    assert judged.score_names == ["judge_hit@5"]
    assert judged.scores["rag"] == {"judge_hit@5": 0.5}
    assert judged.scores["okf"] == {"judge_hit@5": 1.0}


def test_mixed_report_uses_one_shared_metric_column_family():
    run = comparison.EvalRun(
        questions=[],
        relevance=[],
        methods=["rag", "okf"],
        warmup_count=0,
        rankings={},
        timings={
            method: {
                "seconds": 0.0,
                "ms_per_query": 0.0,
                "queries_per_query": 1.0,
                "total_queries": 0.0,
            }
            for method in ("rag", "okf")
        },
        score_names=["hit@1", "hit@5", "recall@5", "mrr@5", "judge_hit@5"],
        scores={
            "rag": {
                "hit@1": 0.0,
                "hit@5": 0.5,
                "recall@5": 0.5,
                "mrr@5": 0.25,
                "judge_hit@5": 0.5,
            },
            "okf": {
                "hit@1": 1.0,
                "hit@5": 1.0,
                "recall@5": 1.0,
                "mrr@5": 1.0,
                "judge_hit@5": 1.0,
            },
        },
    )

    report = comparison.format_eval_report(run, include_categories=False)

    assert "concept_" not in report
    assert report.count("hit@1") == 1
    assert report.count("judge_hit@5") == 1


def test_chunk_checkpoint_can_be_extended_once_with_okf():
    previous_signature = {
        "version": 2,
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
        "methods": {"rag": comparison._empty_method_state()},
    }
    signature = {
        **previous_signature,
        "method_names": ["rag", "okf"],
        "scoring_unit": "mixed",
        "okf_bundle": "/bundle",
        "okf_covered_pages": 2,
        "okf_concepts": 1,
        "okf_methods": ["okf"],
    }

    assert comparison._extend_compatible_state(state, signature, ["rag", "okf"])
    assert state["signature"] == signature
    assert set(state["methods"]) == {"rag", "okf"}


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
        ["rag-a", "rag-b", "okf"],
    ) == {
        "rag-a": 200,
        "rag-b": 199,
        "okf": 199,
    }


def test_equivalence_space_uses_chunks_for_rag_and_concepts_for_okf():
    space = equivalence._EvidenceSpace(
        questions={},
        relevant_ids={},
        documents={},
        chunk_pages={"chunk-a": "page-a", "chunk-b": "page-b"},
        page_concepts={
            "page-a": ["destinations/a.md"],
            "page-b": ["destinations/a.md", "destinations/b.md"],
        },
        okf_methods=frozenset({"okf"}),
        concept_relevant_ids={"q1": ["destinations/a.md"]},
        concept_documents={"destinations/a.md": "Concept evidence"},
    )

    assert space.project_ranking("rag", ["chunk-a", "chunk-b"]) == [
        "chunk-a",
        "chunk-b",
    ]
    assert space.project_ranking("okf", ["destinations/b.md"]) == ["destinations/b.md"]
    assert space.relevant_for("rag", "q1") == []
    assert space.relevant_for("okf", "q1") == ["destinations/a.md"]
    assert space.documents_for("rag") == {}
    assert space.documents_for("okf") == {"destinations/a.md": "Concept evidence"}
