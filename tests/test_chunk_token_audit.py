import json

import pytest

from src.indexing.token_audit import (
    Distribution,
    LanguagePair,
    ProviderTokenizer,
    compare_language_pairs,
    format_corpus_table,
    format_pair_table,
    load_language_pairs,
    measure_text,
    summarize_measurements,
)


class FakeTokenizer:
    model_max_length = 512

    def __call__(self, text, *, add_special_tokens, truncation):
        assert truncation is False
        tokens = text.split()
        if add_special_tokens:
            tokens = ["<s>", *tokens, "</s>"]
        return {"input_ids": list(range(len(tokens)))}


def _runtime(limit=8):
    return ProviderTokenizer(
        provider="fake",
        model_name="fake/model",
        configured_limit=limit,
        effective_limit=limit,
        document_prompt="passage: ",
        tokenizer=FakeTokenizer(),
    )


def test_measure_text_counts_prompt_special_tokens_and_exact_limit():
    measurement = measure_text(
        _runtime(limit=5),
        item_id="b",
        version="v1",
        text="one two",
        canonical_text="one",
        language=None,
    )

    assert measurement.canonical_tokens == 1
    assert measurement.prompt_tokens == 1
    assert measurement.special_tokens == 2
    assert measurement.token_count == 5
    assert measurement.overhead_tokens == 4
    assert measurement.lost_tokens == 0


def test_measure_text_reports_tokens_lost_over_limit():
    measurement = measure_text(
        _runtime(limit=4), item_id="a", version="v2", text="one two"
    )

    assert measurement.token_count == 5
    assert measurement.lost_tokens == 1


def test_grouping_uses_unknown_language_and_deterministic_worst_order():
    rows = [
        measure_text(_runtime(limit=4), item_id=item_id, version="v1", text="one two")
        for item_id in ["b", "a"]
    ]

    summaries = summarize_measurements(rows)

    assert summaries[0].language == "unknown"
    assert summaries[0].over_limit_count == 2
    assert summaries[0].total_lost_tokens == 2
    assert summaries[0].worst_item_ids == ("a", "b")
    assert "a, b" in format_corpus_table(summaries)


def test_language_pairs_measure_source_only_text_and_ratios():
    pairs = [
        LanguagePair(
            id="pair-1",
            sl="ena dve tri",
            en="one two",
            source="fixture",
            reviewed_by="Reviewer",
        )
    ]

    row = compare_language_pairs(_runtime(), pairs)[0]

    assert row.sl_tokens == 3
    assert row.en_tokens == 2
    assert row.token_ratio_sl_to_en == pytest.approx(1.5)


def test_summary_handles_empty_canonical_text_and_interpolates_percentiles():
    rows = [
        measure_text(_runtime(), item_id="empty", version="v1", text=""),
        measure_text(_runtime(), item_id="also-empty", version="v1", text=""),
    ]

    summary = summarize_measurements(rows)[0]

    assert summary.tokens_per_character == 0.0
    assert summary.tokens == Distribution(3.0, 3.0, 3.0, 3.0, 3)


def test_pair_table_uses_interpolated_median():
    pairs = [
        LanguagePair("one", "ena dve", "one", "fixture", "Reviewer"),
        LanguagePair("two", "ena dve tri štiri", "one", "fixture", "Reviewer"),
    ]

    table = format_pair_table(compare_language_pairs(_runtime(), pairs))

    assert "3/3.8/3.9/3.98/4" in table


@pytest.mark.parametrize(
    "rows,error",
    [
        ([{"id": "one"}], "missing"),
        (
            [
                {
                    "id": "one",
                    "sl": "ena",
                    "en": "one",
                    "source": "source",
                    "reviewed_by": "Reviewer",
                },
                {
                    "id": "one",
                    "sl": "dve",
                    "en": "two",
                    "source": "source",
                    "reviewed_by": "Reviewer",
                },
            ],
            "unique",
        ),
    ],
)
def test_language_pair_fixture_rejects_malformed_data(tmp_path, rows, error):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_language_pairs(path, expected_count=len(rows))


def test_language_pair_fixture_requires_expected_count(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "one",
                "sl": "ena",
                "en": "one",
                "source": "source",
                "reviewed_by": "Reviewer",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="30 pairs"):
        load_language_pairs(path)
