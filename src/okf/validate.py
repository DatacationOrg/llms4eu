from __future__ import annotations

import argparse
from pathlib import Path

from src.okf.bundle import validate_bundle
from src.shared.env import ROOT, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / CONFIG["bundle_path"])
    args = parser.parse_args()
    report = validate_bundle(args.bundle)
    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}")
    if not report.valid:
        raise SystemExit(1)
    print(f"valid OKF bundle: {args.bundle}")


if __name__ == "__main__":
    main()
