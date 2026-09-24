from string import Template

from src.shared.env import ROOT

PROMPTS_DIR = ROOT / "prompts"


def render(name: str, **values: object) -> str:
    """Render prompts/<name>.md, replacing every $placeholder with a value.

    Uses string.Template, not str.format, so a prompt can contain literal
    braces (JSON examples, schema snippets) without escaping them. Missing or
    misspelled placeholders raise instead of silently shipping a broken prompt.
    """
    template = Template((PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8"))
    return template.substitute(values).strip()
