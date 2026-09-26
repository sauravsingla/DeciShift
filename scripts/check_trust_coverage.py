#!/usr/bin/env python3
"""Enforce a stronger coverage floor for trust-critical modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRUST_CRITICAL = (
    "decishift/evidence.py",
    "decishift/flow/actions.py",
    "decishift/flow/attribution.py",
    "decishift/flow/validation.py",
    "decishift/attribution/exact.py",
)
DEFAULT_THRESHOLD = 90.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", nargs="?", default="coverage.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    payload = json.loads(Path(args.coverage_json).read_text(encoding="utf-8"))
    files = payload.get("files", {})
    failures: list[str] = []

    for path in TRUST_CRITICAL:
        entry = files.get(path)
        if entry is None:
            failures.append(f"{path}: missing from coverage report")
            continue
        percent = float(entry["summary"]["percent_covered"])
        print(f"{path}: {percent:.2f}%")
        if percent + 1e-12 < args.threshold:
            failures.append(f"{path}: {percent:.2f}% < {args.threshold:.2f}%")

    if failures:
        print("Trust-critical coverage gate failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Trust-critical coverage gate passed at >= {args.threshold:.2f}% per module.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
