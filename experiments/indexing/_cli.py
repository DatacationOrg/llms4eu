from __future__ import annotations

from collections.abc import Callable


def run_cli(main: Callable[[], None]) -> None:
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
