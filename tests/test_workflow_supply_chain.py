from __future__ import annotations

import re
from pathlib import Path


ACTION_USE = re.compile(r"^\s*-\s+uses:\s+([^@\s]+)@([^\s#]+)", re.MULTILINE)
IMMUTABLE_SHA = re.compile(r"^[0-9a-f]{40}$")


def _workflow_files() -> list[Path]:
    files = sorted(Path(".github/workflows").glob("*.yml"))
    files += sorted(Path(".github/workflows").glob("*.yaml"))
    files += sorted(Path("examples/github-actions").glob("*.yml"))
    files += sorted(Path("examples/github-actions").glob("*.yaml"))
    return files


def test_all_third_party_github_actions_are_pinned_to_full_commit_shas():
    violations: list[str] = []
    checked = 0

    for path in _workflow_files():
        text = path.read_text(encoding="utf-8")
        for action, ref in ACTION_USE.findall(text):
            if action.startswith("./"):
                continue
            checked += 1
            if not IMMUTABLE_SHA.fullmatch(ref):
                violations.append(f"{path}: {action}@{ref}")

    assert checked > 0, "No third-party GitHub Action references were discovered"
    assert not violations, (
        "GitHub Actions must use immutable 40-character commit SHAs; moving tags/branches are not allowed:\n"
        + "\n".join(violations)
    )


def test_checkout_does_not_persist_credentials_in_repository_workflows():
    violations: list[str] = []

    for path in _workflow_files():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if "uses: actions/checkout@" not in line:
                continue
            indent = len(line) - len(line.lstrip())
            window: list[str] = []
            for following in lines[index + 1 :]:
                if following.strip() and len(following) - len(following.lstrip()) <= indent:
                    break
                window.append(following)
            if not any("persist-credentials: false" in following for following in window):
                violations.append(str(path))

    assert not violations, (
        "Checkout steps must disable persisted credentials in repository workflows:\n"
        + "\n".join(sorted(set(violations)))
    )
