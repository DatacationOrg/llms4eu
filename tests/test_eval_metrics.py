from src.eval.metrics import bold_best_table, score_rankings
from src.eval.evaluate import (
    EvalRun,
    _resolve_methods,
    add_judge_adjusted_scores,
    format_eval_report,
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
    assert _resolve_methods([]) == ["sparse_rerank", "qwen_hybrid_rerank"]


def test_judge_adjusted_score_is_added_to_overall_table():
    run = EvalRun(
        questions=[],
        relevance=[],
        methods=["qwen_hybrid_rerank"],
        warmup_count=0,
        rankings={},
        timings={
            "qwen_hybrid_rerank": {
                "seconds": 1.0,
                "ms_per_query": 100.0,
                "queries_per_query": 1.0,
                "total_queries": 4.0,
            }
        },
        score_names=["hit@10"],
        scores={"qwen_hybrid_rerank": {"hit@10": 0.5}},
    )

    judged = add_judge_adjusted_scores(
        run,
        {
            "qwen_hybrid_rerank": {
                "questions": 4,
                "strict_hits": 2,
                "equivalent_misses": 1,
            }
        },
        cutoff=10,
    )
    report = format_eval_report(judged, include_categories=False)

    assert judged.scores["qwen_hybrid_rerank"]["judge_hit@10"] == 0.75
    assert "judge_hit@10" in report
