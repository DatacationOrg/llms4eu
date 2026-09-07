"""Language detection contract.

Language is written into the v2 embedding text and every chunk's Chroma
metadata, so a confident wrong answer is worse than an admitted unknown. These
tests pin both halves: the confusable EU pairs are separated, and thin or
boilerplate input refuses to guess.
"""

from __future__ import annotations

import pytest

from src.preprocess.languages import prose_sample
from src.shared.language import (
    EU_LANGUAGE_CODES,
    MIN_DETECTION_CHARS,
    detect_language,
    detect_language_confidence,
)

# Long enough to be a real sample under MIN_DETECTION_CHARS, and chosen from the
# pairs an n-gram detector actually confuses.
SAMPLES = {
    "sl": (
        "Rajhenburški grad stoji nad reko Savo in je bil prvič omenjen leta 895. "
        "Grad so večkrat prezidali, danes pa v njem deluje muzej s stalnimi "
        "razstavami o zgodovini kraja in okolice."
    ),
    "hr": (
        "Dvorac se nalazi iznad rijeke Save i prvi put je spomenut 895. godine. "
        "Dvorac je nekoliko puta pregrađen, a danas u njemu djeluje muzej sa "
        "stalnim izložbama o povijesti mjesta i okolice."
    ),
    "sk": (
        "Hrad sa nachádza nad riekou Sávou a prvýkrát bol spomenutý v roku 895. "
        "Hrad bol viackrát prestavaný a dnes v ňom pôsobí múzeum so stálymi "
        "výstavami o histórii obce a jej okolia."
    ),
    "en": (
        "The castle stands above the river Sava and was first mentioned in 895. "
        "It was rebuilt several times, and today it houses a museum with "
        "permanent exhibitions about the history of the town and its surroundings."
    ),
    "de": (
        "Die Burg steht über dem Fluss Save und wurde erstmals im Jahr 895 "
        "erwähnt. Sie wurde mehrmals umgebaut und beherbergt heute ein Museum "
        "mit Dauerausstellungen zur Geschichte des Ortes und seiner Umgebung."
    ),
    "el": (
        "Το κάστρο βρίσκεται πάνω από τον ποταμό Σάβα και αναφέρθηκε για πρώτη "
        "φορά το 895. Ανακαινίστηκε πολλές φορές και σήμερα στεγάζει μουσείο με "
        "μόνιμες εκθέσεις για την ιστορία της περιοχής."
    ),
    "bg": (
        "Замъкът се намира над река Сава и е споменат за пръв път през 895 "
        "година. Той е преустройван няколко пъти, а днес в него се помещава "
        "музей с постоянни изложби за историята на района."
    ),
}


@pytest.mark.parametrize("code", sorted(SAMPLES))
def test_detects_each_language_including_the_confusable_slavic_trio(code):
    assert detect_language(SAMPLES[code]) == code


def test_confidence_is_reported_for_review():
    code, confidence = detect_language_confidence(SAMPLES["sl"])

    assert code == "sl"
    assert 0.0 < confidence <= 1.0


def test_short_input_falls_back_instead_of_guessing():
    # A 60-character list of Slovenian church names was reported as Croatian
    # with 0.60 confidence before this floor existed.
    short = "Bazilika Lurške Marije, Brestanica\nCerkev sv. Petra Brestanica"

    assert len(short) < MIN_DETECTION_CHARS
    assert detect_language(short) is None
    assert detect_language(short, fallback="sl") == "sl"


def test_empty_input_is_undetermined():
    assert detect_language("") is None
    assert detect_language("   \n  ", fallback="sl") == "sl"


def test_detector_is_restricted_to_the_eu_set():
    # Japanese is outside the candidate set, so it cannot be returned; the
    # detector must not silently widen to all of `Language.all()`.
    detected = detect_language(
        "この城はサヴァ川の上に立っており、895年に初めて言及されました。" * 4
    )

    assert detected is None or detected in EU_LANGUAGE_CODES


def test_prose_sample_strips_markdown_boilerplate():
    markdown = (
        "![pohle](http://www.brestanica.com/uploads/pohle.gif)\n"
        "Cesta na ribnik 3a\n"
        "8280 BrestanicaTel.: 07 49 73 065\n"
        "e-naslov: pohle.m@siol.net\n"
        "| 8 | dvoposteljnih | sob |\n"
        "## Naslov\n"
        "Gostilna stoji ob ribniku sredi Brestanice in ponuja klasično kuhinjo "
        "ter prenocisca za obiskovalce gradu.\n"
    )

    sample = prose_sample(markdown)

    assert "http" not in sample
    assert "siol.net" not in sample
    assert "![" not in sample and "|" not in sample and "#" not in sample
    assert "Gostilna stoji ob ribniku" in sample


def test_prose_sample_drops_number_heavy_listing_lines():
    listing = "Tel.: 07 49 73 065 / 07 49 73 066 / 8280 / 3a / 2014-12-01\n"
    prose = (
        "Grad Rajhenburg je bil zgrajen med letoma 1131 in 1147 in danes v njem "
        "deluje muzej z razstavami.\n"
    )

    sample = prose_sample(listing + prose)

    assert "07 49 73 065" not in sample
    assert "Grad Rajhenburg je bil zgrajen" in sample


def test_prose_sample_respects_its_size_limit():
    markdown = "Grad stoji nad reko Savo in je bil prvic omenjen leta 895. " * 200

    assert len(prose_sample(markdown, limit=500)) <= 600
