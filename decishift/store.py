from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from decishift.diff.compare import ComparisonResult
from decishift.reports.render import render_json, render_markdown, render_terminal


@dataclass
class RunStore:
    root: Path = Path(".decishift/runs")

    def save(self, result: ComparisonResult) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=False)
        result.records.to_csv(path / "records.csv", index=False)
        if result.attribution is not None:
            result.attribution.to_csv(path / "attribution.csv", index=False)
        if result.interactions is not None:
            result.interactions.to_csv(path / "interactions.csv", index=False)
        if result.cohorts is not None:
            result.cohorts.to_csv(path / "cohorts.csv", index=False)
        (path / "summary.json").write_text(render_json(result), encoding="utf-8")
        (path / "report.md").write_text(render_markdown(result), encoding="utf-8")
        (path / "report.txt").write_text(render_terminal(result), encoding="utf-8")
        (path / "meta.json").write_text(json.dumps({"run_id": run_id}, indent=2), encoding="utf-8")
        return run_id

    def _path(self, run_id: str) -> Path:
        path = (self.root / run_id).resolve()
        root = self.root.resolve()
        if root not in path.parents or not path.exists():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        return path

    def report(self, run_id: str, format: str) -> str:
        path = self._path(run_id)
        mapping = {"terminal": "report.txt", "markdown": "report.md", "json": "summary.json"}
        if format not in mapping:
            raise ValueError("format must be terminal, markdown, or json")
        return (path / mapping[format]).read_text(encoding="utf-8")

    def explain(self, run_id: str, record_id: str) -> dict:
        path = self._path(run_id)
        records = pd.read_csv(path / "records.csv", dtype={"record_id": str})
        match = records[records["record_id"] == str(record_id)]
        if match.empty:
            raise KeyError(f"Record id not found: {record_id}")
        payload = match.iloc[0].to_dict()
        attr_path = path / "attribution.csv"
        if attr_path.exists():
            attr = pd.read_csv(attr_path, dtype={"record_id": str})
            payload["component_attribution"] = attr[attr["record_id"] == str(record_id)].to_dict(orient="records")
        else:
            payload["component_attribution"] = "insufficient evidence"
        return payload
