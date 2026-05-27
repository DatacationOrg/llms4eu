from src.eval.metrics import bold_best_table, score_rankings


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
    assert scores["mrr@10"] == 0.25


def test_bold_best_table_highlights_column_winners():
    table = bold_best_table(
        ["method", "hit@5"],
        [["vector", 0.5], ["rerank", 0.75]],
    )

    assert "| vector | 0.500 |" in table
    assert "| rerank | **0.750** |" in table
