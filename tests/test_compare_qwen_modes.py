"""The comparison driver's decisions: what runs, in which groups, and what resumes."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

INDEXING_EXPERIMENTS = Path(__file__).parents[1] / "experiments" / "indexing"
sys.path.insert(0, str(INDEXING_EXPERIMENTS))

driver = importlib.import_module("compare_qwen_modes")


def test_method_groups_resolve_and_mix_with_bare_names():
    names = driver._resolve_methods("agents,qwen4b, geo")
    assert names[0] == "qwen_hybrid_rerank"
    assert "qwen4b" in names and "qwen_hybrid_rerank_geo" in names
    assert len(names) == len(set(names))  # groups overlap; names do not repeat


def test_every_full_grid_method_is_registered():
    driver._resolve_methods("full")
    with pytest.raises(ValueError, match="Unknown methods: nope"):
        driver._resolve_methods("nope")


def test_full_grid_carries_every_axis():
    full = set(driver.METHOD_GROUPS["full"])
    assert {"qwen", "nemotron8b"} <= full  # embedders
    assert {"qwen_hybrid_rerank_geo", "qwen_hybrid_agentic_tools_geo"} <= full  # geo
    assert {"qwen_hybrid_agentic_gptoss", "dci_gptoss"} <= full  # judge LLMs
    assert {"qwen_hybrid_agentic_high", "dci_high"} <= full  # reasoning
    assert "qwen8b_hybrid_rerank_4b" in full  # reranker rung
    # Every agent's paired baseline is in the grid, or the diagnostics stay empty.
    from src.eval.agentic_diagnostics import baseline_for_agent

    for name in full:
        baseline = baseline_for_agent(name)
        if baseline is not None:
            assert baseline in full, (name, baseline)


def test_provider_groups_put_cheap_methods_first_and_one_embedder_per_group():
    groups = driver._provider_groups(
        ["qwen8b_hybrid_rerank", "sparse", "qwen_hybrid_agentic", "dci", "qwen"]
    )
    assert list(groups) == [None, "qwen", "qwen8b"]
    assert groups[None] == ["sparse", "dci"]
    assert groups["qwen"] == ["qwen_hybrid_agentic", "qwen"]


def _signature(**overrides):
    base = {
        "version": driver.STATE_VERSION,
        "variants": ["base"],
        "method_names": ["sparse"],
        "category": None,
        "limit": None,
        "warmup": 0,
        "design": "shared",
        "span_target": "anchor",
        "result_limit": 10,
        "questions": {"base": {"warmup_ids": [], "timed_ids": ["q1", "q2"]}},
        "retrieval": {"reranker_model": "x"},
    }
    return {**base, **overrides}


def test_checkpoint_extends_with_new_methods_and_variants_only():
    state = {
        "signature": _signature(),
        "cells": {"base|sparse": driver._empty_cell("base", "sparse")},
    }
    state["cells"]["base|sparse"]["next_index"] = 2

    wider = _signature(
        variants=["base", "tok512"],
        method_names=["sparse", "qwen"],
        questions={
            "base": {"warmup_ids": [], "timed_ids": ["q1", "q2"]},
            "tok512": {"warmup_ids": [], "timed_ids": ["q1", "q2"]},
        },
    )
    assert driver._extend_compatible_state(
        state, wider, ["base", "tok512"], ["sparse", "qwen"]
    )
    assert set(state["cells"]) == {
        "base|sparse",
        "base|qwen",
        "tok512|sparse",
        "tok512|qwen",
    }
    assert state["cells"]["base|sparse"]["next_index"] == 2  # kept


def test_checkpoint_refuses_a_changed_pipeline_or_question_set():
    state = {
        "signature": _signature(),
        "cells": {"base|sparse": driver._empty_cell("base", "sparse")},
    }
    assert not driver._extend_compatible_state(
        state, _signature(retrieval={"reranker_model": "y"}), ["base"], ["sparse"]
    )
    assert not driver._extend_compatible_state(
        state,
        _signature(questions={"base": {"warmup_ids": [], "timed_ids": ["q1", "q3"]}}),
        ["base"],
        ["sparse"],
    )
    # Dropping a method the checkpoint holds is not an extension either.
    assert not driver._extend_compatible_state(
        state, _signature(method_names=["qwen"]), ["base"], ["qwen"]
    )


def test_retrieval_signature_tracks_the_judge():
    signature = driver._retrieval_signature()
    for key in (
        "agentic_judge_provider",
        "agentic_judges",
        "azure_model",
        "geo_min_candidates",
    ):
        assert key in signature


def test_geo_stats_accumulate_per_question():
    cell = driver._empty_cell("base", "qwen_hybrid_geo")
    before = {
        "resolved": 3,
        "unresolved": 1,
        "widenings": 1,
        "levels": {"nuts3": 2, "radius": 1},
    }
    after = {
        "resolved": 4,
        "unresolved": 1,
        "widenings": 2,
        "levels": {"nuts3": 2, "radius": 1, "nuts2": 1},
    }
    driver._accumulate_geo(cell, before, after)
    assert cell["geo"] == {
        "resolved": 1,
        "unresolved": 0,
        "widenings": 1,
        "levels": {"nuts3": 0, "radius": 0, "nuts2": 1},
    }
    driver._accumulate_geo(cell, None, None)  # a method without geo stats: untouched
    assert cell["geo"]["resolved"] == 1


def test_geo_stats_are_found_through_agent_wrappers():
    stats = SimpleNamespace(resolved=1, unresolved=0, widenings=0, levels={})
    inner = SimpleNamespace(stats=stats)
    wrapped = SimpleNamespace(base_retriever=SimpleNamespace(base_retriever=inner))
    assert driver._geo_stats(wrapped) is stats
    assert driver._geo_stats(SimpleNamespace()) is None


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("qwen", "plain"),
        ("qwen_hybrid_geo", "geo"),
        ("qwen_hybrid_rerank_geo", "rerank"),
        ("qwen_hybrid_agentic_tools_geo", "agentic"),
        ("dci_k50", "dci"),
    ],
)
def test_cost_kind(method, kind):
    assert driver._cost_kind(method) == kind


def test_reranked_cells_are_priced_per_token_and_the_4b_rung_costs_more():
    small = driver._seconds_per_query("qwen_hybrid_rerank", 400)
    big = driver._seconds_per_query("qwen8b_hybrid_rerank_4b", 400)
    assert small == pytest.approx(400 * driver.SECONDS_PER_RERANK_TOKEN)
    assert big > small
    assert (
        driver._seconds_per_query("qwen_hybrid_agentic", 400)
        == driver.SECONDS_PER_QUERY["agentic"]
    )


def test_catch_up_plan_runs_new_cells_to_each_variants_frontier(tmp_path):
    import json

    checkpoint = tmp_path / "c.json"
    state = {
        "signature": _signature(),
        "cells": {
            "base|sparse": {**driver._empty_cell("base", "sparse"), "next_index": 7},
            "base|qwen": {**driver._empty_cell("base", "qwen"), "next_index": 5},
        },
    }
    checkpoint.write_text(json.dumps(state))
    plan = driver._catch_up_plan(checkpoint, ["base"], ["sparse", "qwen", "nemotron"])
    assert plan == {"base|sparse": 7, "base|qwen": 5, "base|nemotron": 5}


def test_checkpoint_view_gives_the_audit_one_variants_legacy_layout():
    equivalence = importlib.import_module("judge_retrieval_equivalence")
    cell = {"rankings": {"1": ["a"]}}
    state = {
        "version": 4,
        "signature": {
            "questions": {"base": {"timed_ids": ["1"]}, "tok512": {"timed_ids": ["1"]}}
        },
        "cells": {"base|sparse": cell, "tok512|sparse": {"rankings": {}}},
    }
    view = equivalence.checkpoint_view(state, "base")
    assert view["signature"]["timed_ids"] == ["1"]
    assert view["methods"]["sparse"] is cell  # shared, so judgments land in the state
    with pytest.raises(ValueError, match="pass --variant"):
        equivalence.checkpoint_view(state)
    legacy = {"signature": {"timed_ids": ["1"]}, "methods": {}}
    assert equivalence.checkpoint_view(legacy) is legacy
