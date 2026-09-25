from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from decishift.cli import app


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
FLOW_CONFIG = REPO_ROOT / "examples" / "triage" / "flow.yaml"
CONTRACT = REPO_ROOT / "examples" / "triage" / "decision-contract.yaml"


def _run_id(output: str) -> str:
    match = re.search(r"^Run ID:\s*(\S+)\s*$", output, flags=re.MULTILINE)
    assert match is not None, output
    return match.group(1)


def test_graph_compare_verify_gate_cli_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    graph = runner.invoke(app, ["graph", str(FLOW_CONFIG)])
    assert graph.exit_code == 0, graph.output
    assert "final_action" in graph.output

    mermaid = runner.invoke(app, ["graph", str(FLOW_CONFIG), "--format", "mermaid"])
    assert mermaid.exit_code == 0, mermaid.output
    assert "%% baseline" in mermaid.output
    assert "%% candidate" in mermaid.output

    compare = runner.invoke(app, ["compare", str(FLOW_CONFIG)])
    assert compare.exit_code == 0, compare.output
    run_id = _run_id(compare.output)

    verify = runner.invoke(app, ["verify", run_id])
    assert verify.exit_code == 0, verify.output
    assert "Evidence integrity verified" in verify.output
    assert "authenticity" in verify.output

    gate = runner.invoke(app, ["gate", run_id, "--contract", str(CONTRACT)])
    assert gate.exit_code == 0, gate.output
    assert "PASS" in gate.output


def test_demo_cli_no_save(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo", "--rows", "25", "--no-save"])
    assert result.exit_code == 0, result.output
    assert "DeciShift" in result.output
    assert not (tmp_path / ".decishift").exists()
