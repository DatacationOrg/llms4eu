import pytest

from src.shared.prompts import PROMPTS_DIR, render


def test_render_substitutes_every_placeholder():
    text = render("geo_place", question="Wie hoch ist die Burg Clam?")

    assert "$" not in text
    assert "Wie hoch ist die Burg Clam?" in text


def test_render_raises_on_a_missing_variable():
    """A silently unrendered $placeholder would ship to the model as literal text."""
    with pytest.raises(KeyError):
        render("geo_place")


@pytest.mark.parametrize("path", sorted(PROMPTS_DIR.glob("*.md")), ids=lambda p: p.stem)
def test_every_prompt_file_is_valid_and_reachable(path):
    from string import Template

    assert Template(path.read_text(encoding="utf-8")).get_identifiers(), (
        f"{path.name} has no placeholders; inline it or add its variables"
    )
