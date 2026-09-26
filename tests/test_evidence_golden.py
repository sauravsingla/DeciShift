from __future__ import annotations

import shutil
from pathlib import Path

from decishift.evidence import verify_evidence_path

FIXTURE = Path(__file__).parent / "fixtures" / "evidence-v2-golden"


def test_golden_v2_evidence_fixture_verifies() -> None:
    result = verify_evidence_path(FIXTURE)
    assert result.passed
    assert result.manifest_status == "PASS"
    assert [(item.artifact, item.status) for item in result.items] == [("summary.json", "PASS")]


def test_golden_v2_evidence_fixture_detects_tampering(tmp_path: Path) -> None:
    copied = tmp_path / "evidence-v2-golden"
    shutil.copytree(FIXTURE, copied)
    (copied / "summary.json").write_text('{"tampered":true}', encoding="utf-8")

    result = verify_evidence_path(copied)
    assert not result.passed
    assert result.manifest_status == "PASS"
    assert [(item.artifact, item.status) for item in result.items] == [("summary.json", "FAIL")]
