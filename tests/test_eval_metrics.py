from src.eval.metrics import (
    bold_best_table,
    score_rankings,
    score_retrieval_efficiency,
    score_span_rankings,
    score_store_share,
    span_coverage,
)
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


# --- character-overlap metrics ---------------------------------------------
#
# The point of these is that chunk-level hit@k rewards a variant for cutting
# large. Each test below fixes retrieval quality and varies only chunk size, so a
# metric that still separates them is measuring size, not retrieval.

ANCHOR = {"q1": ("page-1", 100, 200)}


def test_char_recall_is_indifferent_to_chunk_size():
    """Both chunkings return the whole answer, so both score a full recall."""
    small = score_span_rankings(
        ANCHOR, {"s": ("page-1", 50, 250)}, {"q1": ["s"]}, ks=(1,)
    )
    large = score_span_rankings(
        ANCHOR, {"l": ("page-1", 0, 4000)}, {"q1": ["l"]}, ks=(1,)
    )

    assert small["char_recall@1"] == 1.0
    assert large["char_recall@1"] == 1.0


def test_char_precision_charges_a_large_chunk_for_the_text_it_dragged_in():
    """This is the separation hit@k cannot make: both hit, one wastes context."""
    small = score_span_rankings(
        ANCHOR, {"s": ("page-1", 50, 250)}, {"q1": ["s"]}, ks=(1,)
    )
    large = score_span_rankings(
        ANCHOR, {"l": ("page-1", 0, 4000)}, {"q1": ["l"]}, ks=(1,)
    )

    assert small["char_precision@1"] == 0.5
    assert large["char_precision@1"] == 0.025
    assert small["iou@1"] > large["iou@1"]


def test_partial_overlap_earns_partial_recall():
    # A chunk boundary through the answer is a real, gradual loss; a chunk-level
    # hit would record it as a complete success.
    scores = score_span_rankings(
        ANCHOR, {"c": ("page-1", 150, 900)}, {"q1": ["c"]}, ks=(1,)
    )

    assert scores["char_recall@1"] == 0.5


def test_a_chunk_from_another_page_costs_precision_and_earns_no_recall():
    scores = score_span_rankings(
        ANCHOR,
        {"right": ("page-1", 100, 200), "wrong": ("page-2", 0, 300)},
        {"q1": ["wrong", "right"]},
        ks=(1, 2),
    )

    assert scores["char_recall@1"] == 0.0
    assert scores["char_recall@2"] == 1.0
    assert scores["char_precision@2"] == 100 / 400


def test_overlapping_chunks_are_not_charged_twice_for_shared_text():
    """Otherwise using overlap would read as a precision penalty by itself."""
    scores = score_span_rankings(
        ANCHOR,
        {"a": ("page-1", 0, 150), "b": ("page-1", 100, 250)},
        {"q1": ["a", "b"]},
        ks=(2,),
    )

    # The union spans 0-250, so precision is 100/250 rather than 100/300.
    assert scores["char_precision@2"] == 0.4


def test_a_fixed_character_budget_favours_the_chunking_that_uses_it_better():
    """The comparison that decides the indexing choice.

    Same budget, same answer, same retrieval order. Small chunks place the
    answer-bearing chunk inside the allowance; one oversized chunk does not fit at
    all, and its variant scores nothing.
    """
    small = score_span_rankings(
        ANCHOR,
        {"noise": ("page-1", 500, 900), "answer": ("page-1", 100, 200)},
        {"q1": ["noise", "answer"]},
        ks=(),
        budgets=(600,),
    )
    large = score_span_rankings(
        ANCHOR,
        {"big": ("page-1", 0, 4000)},
        {"q1": ["big"]},
        ks=(),
        budgets=(600,),
    )

    assert small["budget_recall@600"] == 1.0
    assert large["budget_recall@600"] == 0.0


def test_an_oversized_chunk_is_skipped_rather_than_ending_the_fill():
    # Stopping at the first chunk that does not fit would let one long chunk near
    # the top of the ranking decide the score.
    scores = score_span_rankings(
        ANCHOR,
        {"huge": ("page-1", 1000, 9000), "answer": ("page-1", 100, 200)},
        {"q1": ["huge", "answer"]},
        ks=(),
        budgets=(600,),
    )

    assert scores["budget_recall@600"] == 1.0


def test_questions_without_an_anchor_or_a_ranking_are_excluded():
    """Not scored as misses: one has no ground truth, the other was not run."""
    anchors = {
        "anchored_and_run": ("page-1", 100, 200),
        "anchored_not_run": ("page-1", 100, 200),
        "empty_span": ("page-1", 100, 100),
    }
    rankings = {"anchored_and_run": ["c"], "unanchored": ["c"]}
    chunk_spans = {"c": ("page-1", 100, 200)}

    scores = score_span_rankings(anchors, chunk_spans, rankings, ks=(1,))

    assert span_coverage(anchors, rankings) == 1
    assert scores["char_recall@1"] == 1.0


def test_span_metrics_are_zero_rather_than_absent_without_anchors():
    scores = score_span_rankings({}, {}, {"q1": ["c"]}, ks=(1,), budgets=(600,))

    assert scores == {
        "char_recall@1": 0.0,
        "char_precision@1": 0.0,
        "iou@1": 0.0,
        "budget_recall@600": 0.0,
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


# --- cost of the retrieval -------------------------------------------------
#
# The mirror of the tests above. Those fix retrieval quality and vary chunk size
# to show the quality metrics no longer separate them; these fix quality and vary
# size to show the cost metrics *do* — that is the whole point of reporting them.
# Without a cost column a chunking can buy `char_recall` with size and nothing in
# the report says what it paid.

# The same corpus under two cuttings: 20 chunks of 500 characters against 5 of
# 2,000. Same 10,000-character store, same pages, different granularity.
SMALL_STORE = {f"s{index}": ("page-1", index * 500, (index + 1) * 500) for index in range(20)}
LARGE_STORE = {f"l{index}": ("page-1", index * 2000, (index + 1) * 2000) for index in range(5)}


def test_store_share_charges_the_large_cut_for_the_text_it_returns():
    """Identical k, identical answer found, four times the index read."""
    small = score_store_share(SMALL_STORE, {"q1": list(SMALL_STORE)[:4]}, 10_000, ks=(4,))
    large = score_store_share(LARGE_STORE, {"q1": list(LARGE_STORE)[:4]}, 10_000, ks=(4,))

    assert small["store_share@4"] == 20.0, "4 x 500 of a 10,000-char store"
    assert large["store_share@4"] == 80.0, "4 x 2,000 of the same store"


def test_store_share_is_a_share_of_this_variants_own_store():
    """Not of the corpus: a variant is charged against what it actually indexed."""
    scores = score_store_share(SMALL_STORE, {"q1": ["s0"]}, 20_000, ks=(1,))

    assert scores["store_share@1"] == 2.5


def test_an_overlapping_cut_pays_for_the_copies_it_keeps():
    """Unmerged on purpose. Merging the numerator would make overlap free.

    Two chunks covering the same 500 characters twice are two vectors, two
    embeddings and two candidate slots, and the denominator already counts both.
    """
    overlapping = {"a": ("page-1", 0, 1000), "b": ("page-1", 500, 1500)}

    scores = score_store_share(overlapping, {"q1": ["a", "b"]}, 2000, ks=(2,))

    assert scores["store_share@2"] == 100.0, "1,000 + 1,000 of a 2,000-char store"


def test_a_store_with_no_size_scores_zero_rather_than_raising():
    """A variant whose chunks carry no char_count must not end a sweep."""
    assert score_store_share(SMALL_STORE, {"q1": ["s0"]}, 0, ks=(1,)) == {
        "store_share@1": 0.0
    }


def test_efficiency_ranks_the_cheaper_cut_ahead_at_equal_recall():
    """The correction the whole cost section exists for.

    Both cuts return the answer's whole span, so `char_recall` cannot separate
    them — and `char_recall` is the metric the report ranks on. One of them read
    four times the index to do it.
    """
    small = score_retrieval_efficiency({"char_recall@4": 1.0, "store_share@4": 20.0}, ks=(4,))
    large = score_retrieval_efficiency({"char_recall@4": 1.0, "store_share@4": 80.0}, ks=(4,))

    assert small["recall_per_share@4"] == 4 * large["recall_per_share@4"]


def test_efficiency_is_omitted_rather_than_faked_without_span_metrics():
    """A run with no anchors keeps the share column and loses the ratio."""
    assert score_retrieval_efficiency({"store_share@10": 2.0}, ks=(10,)) == {}
    assert score_retrieval_efficiency({"char_recall@10": 0.9}, ks=(10,)) == {}
    # A cut that returned nothing has no ratio either, rather than an infinite one.
    assert score_retrieval_efficiency(
        {"char_recall@10": 0.0, "store_share@10": 0.0}, ks=(10,)
    ) == {}


def test_a_cost_column_bolds_its_smallest_cell():
    """Bolding the largest share would recommend the most expensive cut."""
    table = bold_best_table(
        ["method", "char_recall@10", "store_share@10"],
        [["a", 0.9, 2.0], ["b", 0.8, 0.5]],
        lower_is_better=["store_share@10"],
    )

    assert "**0.900**" in table, "recall still bolds the largest"
    assert "**0.500**" in table, "share bolds the smallest"
    assert "**2.000**" not in table


def test_a_whole_table_of_costs_can_be_declared_at_once():
    """The per-metric sweep tables have variants for headers, not metric names.

    Best is still per column, as in every other table here — the cheapest method
    within a variant, not the cheapest variant. Comparing the columns is the
    reader's job and is what the numbers are printed for.
    """
    table = bold_best_table(
        ["method", "tok256", "tok1024"],
        [["cheap", 0.5, 2.0], ["dear", 0.9, 1.5]],
        lower_is_better=True,
    )

    assert "**0.500**" in table and "**0.900**" not in table
    assert "**1.500**" in table and "**2.000**" not in table
