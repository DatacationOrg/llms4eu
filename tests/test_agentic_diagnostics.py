from src.eval.agentic_diagnostics import (
    baseline_for_agent,
    build_agentic_diagnostics,
    format_agentic_diagnostics,
)


def test_agentic_diagnostics_reports_paired_retry_quality_and_latency():
    questions = [
        {"id": "q1", "question_type": "crosslingual"},
        {"id": "q2", "question_type": "direct_short"},
        {"id": "q3", "question_type": "vague_short"},
        {"id": "q4", "question_type": "vague_short"},
    ]
    relevance = [
        {"question_id": question_id, "chunk_id": f"gold-{question_id}"}
        for question_id in ("q1", "q2", "q3", "q4")
    ]
    rankings = {
        "nemotron_hybrid_rerank_v2": {
            "q1": ["other"],
            "q2": ["gold-q2"],
            "q3": ["other"],
            "q4": ["gold-q4"],
        },
        "nemotron_hybrid_agentic_v2": {
            "q1": ["other", "gold-q1"],
            "q2": ["gold-q2"],
            "q3": ["other"],
            "q4": ["other"],
        },
    }
    method_states = {
        "nemotron_hybrid_agentic_v2": {
            "observations": {
                "q1": {
                    "elapsed_seconds": 0.4,
                    "query_count": 2,
                    "actions": [
                        _action(1, "original", [0.9, 0.4], sufficient=False),
                        _action(2, "translated", [0.8, 0.3], sufficient=True),
                    ],
                },
                "q2": {
                    "elapsed_seconds": 0.1,
                    "query_count": 1,
                    "actions": [_action(1, "q2", [0.95, 0.2], sufficient=True)],
                },
                "q3": {
                    "elapsed_seconds": 0.12,
                    "query_count": 1,
                    "actions": [_action(1, "q3", [0.3, 0.2], sufficient=True)],
                },
                "q4": {
                    "elapsed_seconds": 0.5,
                    "query_count": 2,
                    "actions": [
                        _action(1, "q4", [0.6], sufficient=False),
                        _action(2, "q4", [0.7, 0.2], sufficient=True),
                    ],
                },
            }
        }
    }
    methods = ["nemotron_hybrid_rerank_v2", "nemotron_hybrid_agentic_v2"]

    diagnostics = build_agentic_diagnostics(
        questions=questions,
        relevance=relevance,
        rankings=rankings,
        method_states=method_states,
        method_names=methods,
        cutoff=5,
    )

    summary = diagnostics["summaries"]["nemotron_hybrid_agentic_v2"]
    assert summary["baseline"] == "nemotron_hybrid_rerank_v2"
    assert summary["retried"] == 2
    assert summary["rewritten"] == 1
    assert summary["expanded"] == 1
    assert summary["retry_precision"] == 0.5
    assert summary["retry_recall"] == 0.5
    assert summary["recovered"] == 1
    assert summary["lost"] == 1
    assert summary["retry_ms"] == 450.0
    assert summary["no_retry_ms"] == 110.0
    assert summary["by_category"]["vague_short"]["questions"] == 2

    q1 = diagnostics["details"]["nemotron_hybrid_agentic_v2"][0]
    assert q1["initial_top_score"] == 0.9
    assert q1["initial_top1_top2_margin"] == 0.5
    assert q1["initial_score_spread"] == 0.5
    assert q1["retrieval_ms"] == 20.0
    assert q1["judge_ms"] == 40.0
    assert q1["final_sufficient"] is True

    report = format_agentic_diagnostics(diagnostics)
    assert "retry precision" in report
    assert "1 (25.0%)" not in report
    assert "2 (50.0%)" in report


def test_agent_baseline_mapping_preserves_v2_suffix():
    assert baseline_for_agent("qwen_agentic") == "qwen_rerank"
    assert baseline_for_agent("azure_hybrid_agentic_v2") == "azure_hybrid_rerank_v2"
    assert baseline_for_agent("azure_hybrid_rerank") is None


def _action(
    attempt: int,
    query: str,
    scores: list[float],
    *,
    sufficient: bool,
) -> dict:
    return {
        "attempt": attempt,
        "judge_query": query,
        "chunks": [
            {"id": f"chunk-{index}", "score": score, "text": "text"}
            for index, score in enumerate(scores)
        ],
        "verdict": {"sufficient": sufficient, "reason": "reason"},
        "retrieval_ms": 10.0,
        "judge_ms": 20.0,
        "top_up_ms": 0.0,
    }
