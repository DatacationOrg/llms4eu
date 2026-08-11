from src.eval.metrics import bold_best_table, score_rankings
from src.eval.evaluate import (
    EvalRun,
    _resolve_methods,
    add_judge_adjusted_scores,
    count_chunk_expansions,
    format_eval_report,
    score_eval_rankings,
)


def test_score_rankings_accepts_multiple_relevant_chunks():
    relevance = [
        {"question_id": "q1", "chunk_id": "a"},
        {"question_id": "q1", "chunk_id": "b"},
        {"question_id": "q2", "chunk_id": "c"},
    ]
    rankings = {
        "q1": ["x", "b"],
        "q2": ["x", "y", "z"],
    }

    scores = score_rankings(relevance, rankings, ks=(1, 5))

    assert scores["hit@1"] == 0
    assert scores["hit@5"] == 0.5
    assert scores["recall@5"] == 0.25
    assert scores["mrr@10"] == 0.25

    short_mrr = score_rankings(relevance, rankings, ks=(1,), mrr_k=1)
    assert short_mrr["recall@1"] == 0
    assert short_mrr["mrr@1"] == 0


def test_score_rankings_handles_missing_relevance():
    scores = score_rankings([], {"q1": ["chunk"]}, ks=(1, 5))

    assert scores == {
        "hit@1": 0.0,
        "hit@5": 0.0,
        "recall@5": 0.0,
        "mrr@10": 0.0,
    }


def test_bold_best_table_highlights_column_winners():
    table = bold_best_table(
        ["method", "hit@5"],
        [["vector", 0.5], ["rerank", 0.75]],
    )

    assert "| vector | 0.500     |" in table
    assert "| rerank | **0.750** |" in table
    assert len({line.index("|") for line in table.splitlines()}) == 1
    assert len({line.rindex("|") for line in table.splitlines()}) == 1


def test_eval_default_methods_are_narrow():
    assert _resolve_methods([]) == ["qwen_agentic", "qwen_hybrid_agentic"]


def test_expanded_metrics_only_show_values_for_agentic_methods():
    relevance = [{"question_id": "q1", "chunk_id": "relevant"}]
    rankings = {
        "sparse_rerank": {"q1": ["x"] * 10},
        "qwen_hybrid_agentic": {"q1": ["x"] * 14 + ["relevant"]},
    }

    score_names, scores = score_eval_rankings(
        relevance,
        rankings,
        ["sparse_rerank", "qwen_hybrid_agentic"],
    )

    assert score_names[-3:] == ["hit@15", "recall@15", "mrr@15"]
    assert "hit@15" not in scores["sparse_rerank"]
    assert scores["qwen_hybrid_agentic"]["hit@15"] == 1.0
    assert scores["qwen_hybrid_agentic"]["mrr@15"] == 1 / 15


def test_count_chunk_expansions_ignores_same_limit_reformulations():
    action_log = [
        {"attempt": 1, "chunks": [1] * 5},
        {"attempt": 2, "chunks": [1] * 5},
        {"attempt": 3, "chunks": [1] * 10},
        {"attempt": 1, "chunks": [1] * 5},
        {"attempt": 2, "chunks": [1] * 10},
        {"attempt": 3, "chunks": [1] * 15},
    ]

    assert count_chunk_expansions(action_log) == 3


def test_judge_adjusted_score_is_added_to_overall_table():
    run = EvalRun(
        questions=[],
        relevance=[],
        methods=["agentic"],
        warmup_count=0,
        rankings={},
        timings={
            "agentic": {
                "seconds": 1.0,
                "ms_per_query": 100.0,
                "queries_per_query": 2.0,
                "total_queries": 2.0,
                "chunk_expansions": 1,
            }
        },
        score_names=["hit@10"],
        scores={"agentic": {"hit@10": 0.5}},
    )

    judged = add_judge_adjusted_scores(
        run,
        {
            "agentic": {
                "questions": 4,
                "strict_hits": 2,
                "equivalent_misses": 1,
            }
        },
        cutoff=10,
    )
    report = format_eval_report(judged, include_categories=False)

    assert judged.scores["agentic"]["judge_hit@10"] == 0.75
    assert "judge_hit@10" in report
    assert "chunk expansions" in report
