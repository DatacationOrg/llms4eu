"""Language detection for EU source texts, backed by `lingua`.

Language is not cosmetic here: it is written into the v2 embedding text
(`Source language: …`) and into every chunk's Chroma metadata, so a wrong label
is embedded into the vectors themselves.

Until now it was assigned by fiat — `initialize_page_artifacts_db` stamped every
source with `default_source_language`. That is correct for a single-language
corpus and silently wrong the moment a non-Slovenian page is scraped.

Lingua is used rather than langid/fastText because the confusable pairs in this
corpus are Slovenian/Croatian/Slovak, which is exactly where n-gram detectors
fail and lingua's rule-based filtering wins. The detector is restricted to the
EU set for the same reason: fewer candidates means both higher accuracy and a
smaller model.
"""

from __future__ import annotations

from functools import cache

__all__ = ["EU_LANGUAGE_CODES", "detect_language", "detect_language_confidence"]

# The EU-24 as ISO 639-1, matching `src/eval/generate_dataset.py`.
EU_LANGUAGE_CODES = (
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
)

# Short strings carry too little signal to separate sl/hr/sk reliably; below this
# the caller's configured fallback is more trustworthy than a coin flip. Measured
# on this corpus: a 60-character list of Slovenian church names is reported as
# Croatian with 0.60 confidence, while every page with 120+ characters of prose
# is called correctly.
MIN_DETECTION_CHARS = 120

# Below this confidence the top candidate is not meaningfully ahead of the rest.
MIN_CONFIDENCE = 0.55


def detect_language(text: str, fallback: str | None = None) -> str | None:
    """Best-guess ISO 639-1 code for `text`, or `fallback` when unsure.

    Returns `fallback` rather than guessing on short or low-confidence input, so
    a caller's configured default stays in charge of the ambiguous cases.
    """
    code, confidence = detect_language_confidence(text)
    if code is None or confidence < MIN_CONFIDENCE:
        return fallback
    return code


def detect_language_confidence(text: str) -> tuple[str | None, float]:
    """The detected code and its confidence, for reporting before committing."""
    stripped = (text or "").strip()
    if len(stripped) < MIN_DETECTION_CHARS:
        return None, 0.0
    values = _detector().compute_language_confidence_values(stripped)
    if not values:
        return None, 0.0
    best = values[0]
    return best.language.iso_code_639_1.name.lower(), float(best.value)


@cache
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    wanted = {code.upper() for code in EU_LANGUAGE_CODES}
    languages = [
        language
        for language in Language.all()
        if language.iso_code_639_1.name in wanted
    ]
    # `with_preloaded_language_models` keeps the per-page cost off the first call
    # of a bulk run; detection is otherwise lazy per language.
    return (
        LanguageDetectorBuilder.from_languages(*languages)
        .with_preloaded_language_models()
        .build()
    )
