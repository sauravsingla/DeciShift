#!/usr/bin/env python3
"""Portable graph/compare/verify/gate smoke path for CI platforms."""

from __future__ import annotations

import re
import subprocess
import sys


def run(*args: str) -> str:
    completed = subprocess.run(
        ["decishift", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def main() -> int:
    run("--help")
    run("demo", "--rows", "100", "--no-save")
    run("graph", "examples/triage/flow.yaml")
    output = run("compare", "examples/triage/flow.yaml")
    match = re.search(r"^Run ID:\s*(\S+)\s*$", output, re.MULTILINE)
    if match is None:
        raise SystemExit("Could not parse Run ID from DeciShift comparison output")
    run_id = match.group(1)
    run("verify", run_id)
    run("gate", run_id, "--contract", "examples/triage/decision-contract.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
