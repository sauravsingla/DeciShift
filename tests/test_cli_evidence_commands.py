from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from decishift.cli import app


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
FLOW_CONFIG = REPO_ROOT / "examples" / "triage" / "flow.yaml"


def _run_id(output: str) -> str:
    match = re.search(r"^Run ID:\s*(\S+)\s*$", output, flags=re.MULTILINE)
    assert match is not None, output
    return match.group(1)


def test_saved_evidence_report_explain_and_compare_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    compare = runner.invoke(app, ["compare", str(FLOW_CONFIG)])
    assert compare.exit_code == 0, compare.output
    run_id = _run_id(compare.output)

    terminal = runner.invoke(app, ["report", run_id, "--format", "terminal"])
    assert terminal.exit_code == 0, terminal.output
    assert "DeciShift DecisionFlow comparison" in terminal.output

    markdown = runner.invoke(app, ["report", run_id, "--format", "markdown"])
    assert markdown.exit_code == 0, markdown.output
    assert "DecisionFlow" in markdown.output

    json_report = runner.invoke(app, ["report", run_id, "--format", "json"])
    assert json_report.exit_code == 0, json_report.output
    payload = json.loads(json_report.output)
    assert payload["metadata"]["mode"] == "flow"

    html = runner.invoke(app, ["report", run_id, "--format", "html"])
    assert html.exit_code == 0, html.output
    assert Path(html.output.strip()).exists()

    explained = runner.invoke(app, ["explain", "--run", run_id, "--id", "1"])
    assert explained.exit_code == 0, explained.output
    explanation = json.loads(explained.output)
    assert str(explanation["record_id"]) == "1"

    terminal_compare = runner.invoke(app, ["compare-runs", run_id, run_id])
    assert terminal_compare.exit_code == 0, terminal_compare.output
    assert "delta: +0.00%" in terminal_compare.output

    json_compare = runner.invoke(app, ["compare-runs", run_id, run_id, "--format", "json"])
    assert json_compare.exit_code == 0, json_compare.output
    comparison = json.loads(json_compare.output)
    assert comparison["decision_shift_rate"]["delta"] == 0.0
