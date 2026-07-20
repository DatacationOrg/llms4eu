from __future__ import annotations


def confirm(prompt: str) -> bool:
    """Return True only when the user types exactly yes."""
    response = input(prompt).strip().casefold()
    return response == "yes"
