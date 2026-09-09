"""Sweep-runner logic that decides what gets measured and what gets reported.

The runner itself is glue; these are the parts that can silently mislead: which
cells get pruned, whether an incomparable table is presented as a result, and
whether a method name resolves to the provider whose sequence limit applies.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

INDEXING_EXPERIMENTS = Path(__file__).parents[1] / "experiments" / "indexing"
sys.path.insert(0, str(INDEXING_EXPERIMENTS))

sweep = importlib.import_module("compare_chunkings")


def _audit(max_tokens: int, limits: dict[str, int], models: dict[str, str]) -> dict:
    return {
        "chunks": 100,
        "max_tokens": dict.fromkeys(limits, max_tokens),
        "limits": limits,
        "models": models,
        "median_tokens": dict.fromkeys(limits, max_tokens // 2),
        "over_limit": dict.fromkeys(limits, 0.0),
    }


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("qwen", "qwen"),
        ("qwen_hybrid_rerank", "qwen"),
        ("qwen_s512", "qwen_s512"),
        ("qwen_s512_hybrid_rerank", "qwen_s512"),
        ("qwen4b_rerank", "qwen4b"),
        ("sparse", ""),
    ],
)
def test_method_provider_prefers_the_longest_matching_provider(method, expected):
    # "qwen_s512_hybrid" starts with "qwen_" too; the longest match is the
    # provider whose sequence limit actually applies to that cell.
    assert sweep._method_provider(method) == expected


def test_fit_leaves_a_provider_alone_when_it_already_reads_the_variant_whole():
    audit = {
        "base": _audit(
            1215, {"qwen": 32768, "qwen_s512": 512}, {"qwen": "Q", "qwen_s512": "Q"}
        )
    }

    assert sweep.fit_provider("base", "qwen", audit) == "qwen"


def test_fit_swaps_in_the_shortest_sequence_length_that_reads_the_variant():
    # 1215 tokens overflows 512 but fits 32768, so measuring `base` on the 512
    # provider would report truncation as if it were a chunk-size effect.
    audit = {
        "base": _audit(
            1215, {"qwen": 32768, "qwen_s512": 512}, {"qwen": "Q", "qwen_s512": "Q"}
        )
    }

    assert sweep.fit_provider("base", "qwen_s512", audit) == "qwen"


def test_fit_never_crosses_model_families():
    audit = {
        "base": _audit(
            1215, {"english": 256, "qwen": 32768}, {"english": "MiniLM", "qwen": "Q"}
        )
    }

    # No MiniLM sibling can read it, so the caller must report the shortfall
    # rather than silently substituting a different model.
    assert sweep.fit_provider("base", "english", audit) == "english"


def test_refit_rewrites_only_the_provider_prefix_of_a_method():
    assert sweep._refit("qwen_s512_hybrid_rerank", "qwen_s512", "qwen") == (
        "qwen_hybrid_rerank"
    )
    assert sweep._refit("qwen_s512", "qwen_s512", "qwen") == "qwen"


def test_a_variant_no_provider_can_read_is_skipped_with_a_reason(monkeypatch):
    monkeypatch.setattr(sweep, "missing_retriever_indexes", lambda *_: {})
    audit = {"base": _audit(1215, {"english": 256}, {"english": "MiniLM"})}

    planned, skipped, refitted = sweep._plan(["base"], ["english"], audit)

    assert planned == []
    assert refitted == []
    assert "no configured sequence length fits" in skipped[0]


def test_no_fit_keeps_a_truncating_cell_so_the_old_cap_stays_measurable(monkeypatch):
    monkeypatch.setattr(sweep, "missing_retriever_indexes", lambda *_: {})
    audit = {
        "base": _audit(
            1253,
            {"qwen": 32768, "qwen_s512": 512},
            {
                "qwen": "Qwen/Qwen3-Embedding-0.6B",
                "qwen_s512": "Qwen/Qwen3-Embedding-0.6B",
            },
        )
    }

    planned, skipped, refitted = sweep._plan(
        ["base"], ["qwen", "qwen_s512"], audit, fit=False
    )

    # --no-fit is how the cost of the historical 512 cap is measured: the same
    # chunks and labels read at two different lengths.
    assert planned == [("base", "qwen"), ("base", "qwen_s512")]
    assert skipped == [] and refitted == []


def test_fit_on_rewrites_the_truncating_cell_instead_of_measuring_truncation(
    monkeypatch,
):
    monkeypatch.setattr(sweep, "missing_retriever_indexes", lambda *_: {})
    audit = {
        "base": _audit(
            1253,
            {"qwen": 32768, "qwen_s512": 512},
            {
                "qwen": "Qwen/Qwen3-Embedding-0.6B",
                "qwen_s512": "Qwen/Qwen3-Embedding-0.6B",
            },
        )
    }

    planned, _, refitted = sweep._plan(["base"], ["qwen_s512"], audit, fit=True)

    assert planned == [("base", "qwen")]
    assert len(refitted) == 1 and "reads it whole" in refitted[0]


def test_different_models_are_never_pruned_against_each_other(monkeypatch):
    monkeypatch.setattr(sweep, "missing_retriever_indexes", lambda *_: {})
    audit = {
        "tok256": _audit(
            300,
            {"qwen": 512, "nemotron": 4096},
            {
                "qwen": "Qwen/Qwen3-Embedding-0.6B",
                "nemotron": "nvidia/Nemotron-3-Embed-1B-BF16",
            },
        )
    }

    planned, _, _ = sweep._plan(["tok256"], ["qwen", "nemotron"], audit)

    assert planned == [("tok256", "qwen"), ("tok256", "nemotron")]


def test_a_missing_index_is_skipped_and_named(monkeypatch):
    monkeypatch.setattr(
        sweep,
        "missing_retriever_indexes",
        lambda names, variant: {names[0]: "qwen#tok9"},
    )
    audit = {"tok9": _audit(300, {"qwen": 512}, {"qwen": "m"})}

    planned, skipped, _ = sweep._plan(["tok9"], ["qwen"], audit)

    assert planned == []
    assert skipped == ["tok9|qwen: index not built (qwen#tok9)"]


def test_recommendation_refuses_unequal_samples_on_a_shared_target_metric():
    """Span metrics score one shared target set, so unequal samples are a fault.

    0.95 over 200 targets is not a win over 0.80 over 3,476: it means the targets
    were never finished, not that the variant is better.
    """
    cells = [
        sweep.Cell(
            "base",
            "qwen",
            {"budget_recall@4000": 0.80},
            questions=3476,
            span_questions=3476,
        ),
        sweep.Cell(
            "tok512",
            "qwen",
            {"budget_recall@4000": 0.95},
            questions=200,
            span_questions=200,
        ),
    ]

    text = sweep._recommendation(cells, ["base", "tok512"], {})

    assert "not comparable" in text
    assert "0.95" not in text


def test_recommendation_accepts_unequal_question_counts_per_variant():
    """Under per-variant questions the counts differ by design, not by omission.

    Each variant's question set is sized by its own chunk count, and every
    question is scored independently, so the comparison holds — the smaller
    samples are noisier rather than biased.
    """
    cells = [
        sweep.Cell("tok1024", "qwen", {"recall@10": 0.80}, questions=1410),
        sweep.Cell("tok256", "qwen", {"recall@10": 0.86}, questions=7647),
    ]

    text = sweep._recommendation(cells, ["tok1024", "tok256"], {}, "per-variant")

    assert "not comparable" not in text
    assert "0.860" in text, "a winner is still named"
    assert "noisier" in text, "and the uneven samples are still disclosed"


def test_unequal_counts_under_the_shared_design_are_reported_as_incomplete():
    """The same spread means the opposite thing once the questions are shared.

    Under `per-variant` differing counts are the design; under `shared` one
    question set is projected onto every cutting, so a short column means that
    variant has chunks with no resolved span. Calling it "noisier" there would
    excuse missing labels as sampling noise.
    """
    cells = [
        sweep.Cell("tok1024", "qwen", {"recall@10": 0.80}, questions=1410),
        sweep.Cell("tok256", "qwen", {"recall@10": 0.86}, questions=7647),
    ]

    text = sweep._recommendation(cells, ["tok1024", "tok256"], {}, "shared")

    assert "noisier" not in text
    assert "incomplete" in text


def test_recommendation_names_the_winner_and_its_truncation():
    cells = [
        sweep.Cell("base", "qwen", {"recall@10": 0.700}, questions=500),
        sweep.Cell("tok512", "qwen", {"recall@10": 0.812}, questions=500),
    ]
    audit = {
        "base": _audit(1253, {"qwen": 512}, {"qwen": "m"}),
        "tok512": _audit(494, {"qwen": 512}, {"qwen": "m"}),
    }
    audit["base"]["over_limit"]["qwen"] = 0.66

    text = sweep._recommendation(cells, ["base", "tok512"], audit)

    assert "tok512" in text and "0.812" in text
    assert "+0.112" in text


def test_cells_without_a_metric_render_as_a_dash_not_zero():
    assert sweep._cell_value(None, "recall@10") == "-"
    assert sweep._cell_value(sweep.Cell("v", "m", {}), "recall@10") == "-"


def test_a_present_metric_reaches_the_table_as_a_number():
    """Stringifying it here is what disabled bolding in every metric section.

    `bold_best_table` formats floats to the same three decimals itself and can
    only bold what it can compare, so no published sweep report has a bold cell
    in one of these tables. It matters now that a cost metric sits beside the
    quality ones and the bold has to be able to land on the smallest cell.
    """
    value = sweep._cell_value(sweep.Cell("v", "m", {"recall@10": 0.5}), "recall@10")

    assert value == 0.5 and isinstance(value, float)
    assert "**0.500**" in sweep.bold_best_table(["method", "v"], [["m", value]])


def test_dedup_never_collapses_different_retrieval_methods(monkeypatch):
    monkeypatch.setattr(sweep, "missing_retriever_indexes", lambda *_: {})
    audit = {"base": _audit(400, {"qwen": 32768}, {"qwen": "Q"})}

    planned, skipped, _ = sweep._plan(
        ["base"], ["qwen", "qwen_hybrid", "qwen_hybrid_rerank"], audit
    )

    # These are three methods on one provider, not three lengths of one method.
    # Keying the duplicate check on (variant, model) alone left only the first.
    assert planned == [
        ("base", "qwen"),
        ("base", "qwen_hybrid"),
        ("base", "qwen_hybrid_rerank"),
    ]
    assert skipped == []


def test_dedup_still_collapses_two_lengths_of_the_same_method(monkeypatch):
    monkeypatch.setattr(sweep, "missing_retriever_indexes", lambda *_: {})
    audit = {
        "tok256": _audit(
            300, {"qwen": 32768, "qwen_s512": 512}, {"qwen": "Q", "qwen_s512": "Q"}
        )
    }

    planned, skipped, _ = sweep._plan(
        ["tok256"], ["qwen_hybrid", "qwen_s512_hybrid"], audit
    )

    # 300 tokens fits both limits, so the vectors are identical.
    assert planned == [("tok256", "qwen_hybrid")]
    assert len(skipped) == 1 and "same retrieval method" in skipped[0]


class _Args:
    """The subset of parsed arguments the report and cost paths actually read."""

    def __init__(self, **overrides):
        self.design = "shared"
        self.span_target = "gold"
        self.span_metrics = True
        self.limit = None
        self.category = None
        self.warmup = 0
        self.judge_equivalence = False
        self.judge_k = 10
        self.agentic_diagnostics = False
        self.__dict__.update(overrides)


def _cell(variant, method, **overrides):
    cell = sweep.Cell(variant=variant, method=method)
    for name, value in overrides.items():
        setattr(cell, name, value)
    return cell


def test_named_groups_expand_and_mix_with_bare_method_names():
    resolved = sweep._resolve_methods("sweep-completion-cheap,sparse")
    assert resolved == ["nemotron_hybrid", "nemotron_hybrid_rerank", "sparse"]


def test_group_expansion_never_duplicates_a_method_two_groups_share():
    # `nemotron_hybrid_rerank` is in both groups. A duplicate would run the same
    # cell twice and put two identical rows in the table.
    resolved = sweep._resolve_methods("symmetric,agentic")
    assert len(resolved) == len(set(resolved))
    assert resolved.count("nemotron_hybrid_rerank") == 1


# --- question-set design ----------------------------------------------------
#
# Sample size alone stopped identifying the design once `--density` existed: its
# whole point is per-variant question sets of matching size, which is exactly what
# the old check called "shared". The stubs below therefore carry question *ids*,
# because whether the same question appears in two variants is the observable that
# stays structural — a per-variant question id derives from a chunk id, and chunk
# ids carry the variant.


def _stub_labels(monkeypatch, labels: dict[str, list[str]]) -> None:
    monkeypatch.setattr(
        sweep,
        "load_eval_rows",
        lambda variant, **_: ([{"id": qid} for qid in labels[variant]], []),
    )


def _shared(count: int) -> list[str]:
    """One question set, projected onto every variant: identical ids."""
    return [f"q{index}" for index in range(count)]


def _own(variant: str, count: int) -> list[str]:
    """A variant's own questions: ids no other variant can carry."""
    return [f"{variant}:q{index}" for index in range(count)]


def test_shared_design_refuses_variants_labelled_for_different_samples(monkeypatch):
    # The failure this exists to prevent: per-variant questions scored as though
    # one shared set had been projected onto every cutting.
    _stub_labels(
        monkeypatch, {"base": _own("base", 3476), "tok256": _own("tok256", 7806)}
    )
    with pytest.raises(RuntimeError, match="use --design per-variant"):
        sweep._check_design(["base", "tok256"], "shared")


def test_per_variant_design_refuses_a_shared_question_set(monkeypatch):
    _stub_labels(monkeypatch, {"base": _shared(3476), "tok256": _shared(3476)})
    with pytest.raises(RuntimeError, match="use --design shared"):
        sweep._check_design(["base", "tok256"], "per-variant")


def test_shared_design_accepts_a_sample_within_the_allowed_spread(monkeypatch):
    # One question dropped because a variant chunk has no resolved span is not a
    # reason to refuse the table.
    _stub_labels(monkeypatch, {"base": _shared(3476), "tok256": _shared(3476)[:-1]})
    sweep._check_design(["base", "tok256"], "shared")


def test_density_normalised_sets_are_not_mistaken_for_a_shared_one(monkeypatch):
    """The check the old sample-size test would have got backwards.

    Under `--density` each variant owns its questions *and* the counts match, so
    counting rows says "shared" while the ids say otherwise. Getting this wrong
    publishes per-variant columns under a shared-design heading, which is the one
    error a reader cannot see in the finished table.
    """
    _stub_labels(
        monkeypatch, {"tok256": _own("tok256", 2003), "tok1024": _own("tok1024", 2004)}
    )

    sweep._check_design(["tok256", "tok1024"], "per-variant-density")

    with pytest.raises(RuntimeError, match="per-variant-density if they were"):
        sweep._check_design(["tok256", "tok1024"], "shared")


def test_the_two_per_variant_designs_are_told_apart_by_sample_size(monkeypatch):
    """Density mode's promise is equal counts, so unequal counts mean it did not run."""
    _stub_labels(
        monkeypatch, {"tok256": _own("tok256", 7801), "tok1024": _own("tok1024", 1451)}
    )

    sweep._check_design(["tok256", "tok1024"], "per-variant")

    with pytest.raises(RuntimeError, match="legacy one-question-per-type"):
        sweep._check_design(["tok256", "tok1024"], "per-variant-density")


def test_legacy_per_variant_counts_are_refused_as_density_normalised(monkeypatch):
    _stub_labels(
        monkeypatch, {"tok256": _own("tok256", 2003), "tok1024": _own("tok1024", 2004)}
    )
    with pytest.raises(RuntimeError, match="use --design per-variant-density"):
        sweep._check_design(["tok256", "tok1024"], "per-variant")


def test_an_unlabelled_variant_is_named_with_the_command_that_labels_it(monkeypatch):
    _stub_labels(monkeypatch, {"base": _shared(3476), "tok512": []})
    with pytest.raises(RuntimeError, match="src.eval.relabel"):
        sweep._check_design(["base", "tok512"], "shared")


def test_an_unlabelled_variant_under_density_is_told_to_use_the_flag(monkeypatch):
    _stub_labels(monkeypatch, {"base": _own("base", 3476), "tok512": []})
    with pytest.raises(RuntimeError, match="--density"):
        sweep._check_design(["base", "tok512"], "per-variant-density")


def test_a_rerank_method_without_a_provider_is_not_priced_as_free():
    # `sparse_rerank` carries no provider name but runs the same cross-encoder, so
    # pricing it from a missing median would quote a 40-minute cell as free.
    audit = {
        "base": _audit(1200, {"qwen": 32768}, {"qwen": "Qwen/Qwen3-Embedding-0.6B"})
    }
    assert sweep._median_tokens("base", "sparse_rerank", audit) > 0
    assert sweep._cost_kind("sparse_rerank") == "rerank"


def test_reranker_input_limit_caps_the_priced_chunk_length():
    # Past the reranker's own max_length the text is truncated and costs no more.
    limit = sweep.RETRIEVAL_CONFIG["reranker_max_length"]
    audit = {"base": _audit(limit * 8, {"qwen": 32768}, {"qwen": "m"})}
    assert sweep._median_tokens("base", "qwen_hybrid_rerank", audit) == limit


def test_cost_kinds_separate_the_three_rates():
    assert sweep._cost_kind("qwen") == "plain"
    assert sweep._cost_kind("qwen_hybrid_rerank") == "rerank"
    assert sweep._cost_kind("nemotron_hybrid_agentic") == "agentic"
    assert sweep._cost_kind("dci_k50") == "agentic"


def test_diagnostics_run_the_baseline_in_the_same_cell_only_when_asked():
    # The paired table needs both in one run, and that extra baseline is a real
    # per-variant cost, so it must not appear without --agentic-diagnostics.
    plain = _Args()
    diagnosed = _Args(agentic_diagnostics=True)
    assert sweep._cell_methods("nemotron_hybrid_agentic", plain) == [
        "nemotron_hybrid_agentic"
    ]
    assert sweep._cell_methods("nemotron_hybrid_agentic", diagnosed) == [
        "nemotron_hybrid_agentic",
        "nemotron_hybrid_rerank",
    ]
    # A method with no agentic baseline stays a single-method cell either way.
    assert sweep._cell_methods("qwen_hybrid_rerank", diagnosed) == [
        "qwen_hybrid_rerank"
    ]


def test_speed_is_reported_per_query_not_as_a_total(monkeypatch):
    # Totals would rank the smallest question set fastest and say nothing about
    # the method, which is the whole reason this column is a rate.
    cells = [
        _cell("base", "qwen", ms_per_query=8.0, seconds=400.0),
        _cell("tok1024", "qwen", ms_per_query=9.0, seconds=40.0),
    ]
    by_key = {(cell.variant, cell.method): cell for cell in cells}
    section = "\n".join(sweep._speed_section(["base", "tok1024"], ["qwen"], by_key))
    assert "8.0" in section and "9.0" in section
    assert "400" not in section


def test_a_cell_that_produced_no_timing_is_left_out_of_speed():
    by_key = {("base", "qwen"): _cell("base", "qwen", ms_per_query=0.0)}
    assert sweep._speed_section(["base"], ["qwen"], by_key) == []


def test_category_tables_use_one_section_per_question_type():
    cells = [
        _cell(
            "base", "qwen", category_scores={"crosslingual": 0.5, "vague_short": 0.3}
        ),
        _cell(
            "tok512", "qwen", category_scores={"crosslingual": 0.9, "vague_short": 0.4}
        ),
    ]
    by_key = {(cell.variant, cell.method): cell for cell in cells}
    section = "\n".join(
        sweep._category_sections(["base", "tok512"], ["qwen"], by_key, cells)
    )
    assert "### crosslingual" in section
    assert "### vague_short" in section
    assert "0.900" in section


def test_cell_rows_are_written_long_with_each_cells_own_sample_size(tmp_path):
    # Merging with another report joins on (variant, method, metric); a merged
    # table must not be able to hide that two rows had different samples.
    cells = [
        _cell(
            "base",
            "qwen",
            scores={"hit@5": 0.5},
            category_scores={"crosslingual": 0.25},
            questions=3476,
            span_questions=3476,
            ms_per_query=8.0,
        ),
        _cell(
            "tok256", "qwen", scores={"hit@5": 0.6}, questions=7806, ms_per_query=9.0
        ),
    ]
    path = sweep._write_cell_rows(
        tmp_path / "out.csv", cells, _Args(design="per-variant")
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == [
        "design",
        "variant",
        "method",
        "metric",
        "value",
        "questions",
        "span_questions",
        "ms_per_query",
    ]
    assert "per-variant,base,qwen,hit@5,0.500000,3476,3476,8.000" in lines
    assert (
        "per-variant,base,qwen,category_hit/crosslingual,0.250000,3476,3476,8.000"
        in lines
    )
    assert "per-variant,tok256,qwen,hit@5,0.600000,7806,0,9.000" in lines


def test_the_report_states_which_design_and_span_target_produced_it():
    cells = [_cell("base", "qwen", scores={"recall@10": 0.9}, questions=10)]
    audit = {"base": _audit(600, {"qwen": 32768}, {"qwen": "m"})}
    report = sweep._format_report(
        cells=cells,
        variants=["base"],
        methods=["qwen"],
        audit=audit,
        skipped=[],
        refitted=[],
        command="x",
        args=_Args(design="per-variant", span_target="anchor"),
    )
    assert "Question-set design: **per-variant**" in report
    assert "Span target: **anchor**" in report


def test_gold_target_report_says_base_is_the_ruler_not_a_competitor():
    cells = [_cell("base", "qwen", scores={"recall@10": 0.9}, questions=10)]
    audit = {"base": _audit(600, {"qwen": 32768}, {"qwen": "m"})}
    report = sweep._format_report(
        cells=cells,
        variants=["base"],
        methods=["qwen"],
        audit=audit,
        skipped=[],
        refitted=[],
        command="x",
        args=_Args(span_target="gold"),
    )
    assert "ruler" in report


def _write_checkpoint(path, signature, runs):
    import json

    path.write_text(
        json.dumps({"signature": signature, "runs": runs}), encoding="utf-8"
    )


def test_reranker_depth_is_part_of_the_checkpoint_signature():
    # Halving rerank_candidate_limit halves the reranker's work and changes every
    # score it produces. A resume that ignored it would put cells measured at two
    # depths in one table, where the gap reads as a chunking difference.
    assert "rerank_candidate_limit" in sweep._retrieval_signature()
    assert "reranker_model" in sweep._retrieval_signature()
    assert "hybrid_vector_weight" in sweep._retrieval_signature()


def test_changed_retrieval_settings_discard_the_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "c.json"
    args = _Args(no_resume=False, span_metrics=False)
    stored = {
        "variants": ["base"],
        "methods": ["qwen_hybrid_rerank"],
        "limit": None,
        "category": None,
        "warmup": 0,
        "design": "shared",
        "span_target": None,
        "judge_k": None,
        "agentic_diagnostics": False,
        "retrieval": {**sweep._retrieval_signature(), "rerank_candidate_limit": 30},
    }
    _write_checkpoint(checkpoint, stored, {"base|qwen_hybrid_rerank": {"seconds": 1.0}})

    monkeypatch.setitem(sweep.RETRIEVAL_CONFIG, "rerank_candidate_limit", 15)
    state = sweep._load_state(checkpoint, ["base"], ["qwen_hybrid_rerank"], args)

    assert state["runs"] == {}, "cells measured at another rerank depth are not reused"


def test_unchanged_retrieval_settings_keep_the_checkpoint(tmp_path):
    checkpoint = tmp_path / "c.json"
    args = _Args(no_resume=False, span_metrics=False)
    stored = {
        "variants": ["base"],
        "methods": ["qwen_hybrid_rerank"],
        "limit": None,
        "category": None,
        "warmup": 0,
        "design": "shared",
        "span_target": None,
        "judge_k": None,
        "agentic_diagnostics": False,
        "retrieval": sweep._retrieval_signature(),
    }
    _write_checkpoint(checkpoint, stored, {"base|qwen_hybrid_rerank": {"seconds": 1.0}})

    state = sweep._load_state(checkpoint, ["base"], ["qwen_hybrid_rerank"], args)

    assert list(state["runs"]) == ["base|qwen_hybrid_rerank"]


def test_a_checkpoint_predating_retrieval_tracking_resumes_with_a_warning(
    tmp_path, capsys
):
    """Hours of completed cells are not thrown away over an unrecorded field.

    But the guard cannot do its job for those cells, so the run says so instead of
    presenting the resume as verified.
    """
    checkpoint = tmp_path / "c.json"
    args = _Args(no_resume=False, span_metrics=False)
    stored = {
        "variants": ["base"],
        "methods": ["qwen_hybrid_rerank"],
        "limit": None,
        "category": None,
        "warmup": 0,
        "design": "shared",
        "span_target": None,
        "judge_k": None,
        "agentic_diagnostics": False,
    }  # no "retrieval" key: written before the setting was tracked
    _write_checkpoint(checkpoint, stored, {"base|qwen_hybrid_rerank": {"seconds": 1.0}})

    state = sweep._load_state(checkpoint, ["base"], ["qwen_hybrid_rerank"], args)

    assert list(state["runs"]) == ["base|qwen_hybrid_rerank"], "work is preserved"
    out = capsys.readouterr().out
    assert "did not record retrieval settings" in out
    assert "--no-resume" in out, "and the way to be certain is named"


def test_a_discarded_checkpoint_names_which_field_changed(tmp_path, capsys):
    checkpoint = tmp_path / "c.json"
    args = _Args(no_resume=False, span_metrics=False, design="shared")
    stored = {
        "variants": ["base"],
        "methods": ["qwen"],
        "limit": None,
        "category": None,
        "warmup": 0,
        "design": "per-variant",
        "span_target": None,
        "judge_k": None,
        "agentic_diagnostics": False,
        "retrieval": sweep._retrieval_signature(),
    }
    _write_checkpoint(checkpoint, stored, {"base|qwen": {"seconds": 1.0}})

    sweep._load_state(checkpoint, ["base"], ["qwen"], args)

    assert "design" in capsys.readouterr().out


class _Run:
    def __init__(self, methods, questions, relevance, rankings, action_logs):
        self.methods = methods
        self.questions = questions
        self.relevance = relevance
        self.rankings = rankings
        self.action_logs = action_logs


def _action(attempt: int, query: str, chunk_ids: list[str], sufficient: bool) -> dict:
    return {
        "original_query": "q",
        "attempt": attempt,
        "judge_query": query,
        "chunks": [{"id": c, "score": 1.0, "text": ""} for c in chunk_ids],
        "verdict": {"sufficient": sufficient, "reason": "", "reformulated_query": None},
        "retrieval_ms": 1.0,
        "judge_ms": 1.0,
        "top_up_ms": 0.0,
    }


def test_cell_diagnostics_see_the_agent_actions_not_an_empty_log():
    """Retries and rewrites must reach the report; a 0 there must mean inaction."""
    questions = [
        {"id": "q1", "question_type": "factual"},
        {"id": "q2", "question_type": "factual"},
    ]
    relevance = [
        {"question_id": "q1", "chunk_id": "a"},
        {"question_id": "q2", "chunk_id": "b"},
    ]
    rankings = {
        "qwen_hybrid_rerank": {"q1": ["x", "a"], "q2": ["b"]},
        "qwen_hybrid_agentic": {"q1": ["a"], "q2": ["b"]},
    }
    # q1 retried once with a rewritten query; q2 was sufficient at once.
    log = [
        _action(1, "q1 original", ["x", "a"], False),
        _action(2, "q1 rewritten", ["x", "a", "y"], True),
        _action(1, "q2", ["b", "c"], True),
    ]
    run = _Run(
        ["qwen_hybrid_rerank", "qwen_hybrid_agentic"],
        questions,
        relevance,
        rankings,
        {"qwen_hybrid_agentic": log},
    )

    summary = sweep._diagnostics(run, "qwen_hybrid_agentic")["summaries"][
        "qwen_hybrid_agentic"
    ]

    assert summary["questions"] == 2
    assert summary["retried"] == 1
    assert summary["rewritten"] == 1
