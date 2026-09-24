from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from decishift.attribution.diagnostics import AttributionDiagnostics
from decishift.diff.compare import ComparisonResult
from decishift.evidence import build_manifest, sha256_file, verify_evidence_path
from decishift.reports.render import render_html, render_json, render_markdown, render_terminal
from decishift.version import __version__

_EMPTY_ATTR = ["record_id", "component", "score_contribution", "decision_contribution"]
_EMPTY_INTERACTIONS = ["record_id", "component_a", "component_b", "score_interaction", "decision_interaction", "interaction_only_flip"]
_EMPTY_COHORTS = ["dimension", "cohort", "size", "coverage", "flip_rate"]


@dataclass
class RunStore:
    root: Path = Path(".decishift/runs")

    def save(self, result: ComparisonResult) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=False)
        result.records.to_csv(path / "records.csv", index=False)
        (result.attribution if result.attribution is not None else pd.DataFrame(columns=_EMPTY_ATTR)).to_csv(path / "attribution.csv", index=False)
        (result.interactions if result.interactions is not None else pd.DataFrame(columns=_EMPTY_INTERACTIONS)).to_csv(path / "interactions.csv", index=False)
        (result.cohorts if result.cohorts is not None else pd.DataFrame(columns=_EMPTY_COHORTS)).to_csv(path / "cohorts.csv", index=False)
        if result.fragility is not None and isinstance(result.fragility.get("records"), pd.DataFrame):
            result.fragility["records"].to_csv(path / "fragility.csv", index=False)
        if result.outcome_analysis is not None:
            (path / "outcome.json").write_text(json.dumps(result.outcome_analysis, indent=2, allow_nan=False, default=str), encoding="utf-8")
        (path / "summary.json").write_text(render_json(result), encoding="utf-8")
        (path / "report.md").write_text(render_markdown(result), encoding="utf-8")
        (path / "report.txt").write_text(render_terminal(result), encoding="utf-8")
        (path / "report.html").write_text(render_html(result), encoding="utf-8")
        artifact_names = ["summary.json", "records.csv", "attribution.csv", "interactions.csv", "cohorts.csv", "report.md", "report.txt", "report.html"]
        for optional in ("fragility.csv", "outcome.json"):
            if (path / optional).exists():
                artifact_names.append(optional)
        hashes = {name: sha256_file(path / name) for name in sorted(artifact_names)}
        manifest = build_manifest(
            decishift_version=__version__,
            attribution_method=result.metadata.get("attribution_method"),
            attribution_parameters=result.metadata.get("attribution_parameters") or {},
            random_seed=result.metadata.get("random_seed"),
            number_of_records=len(result.records),
            changed_components=result.changed_components,
            baseline_component_identities=result.metadata.get("baseline_component_identities"),
            candidate_component_identities=result.metadata.get("candidate_component_identities"),
            configuration_sha256=result.metadata.get("configuration_sha256"),
            input_data_fingerprint=result.metadata.get("input_data_fingerprint"),
            input_hashing_enabled=bool(result.metadata.get("input_content_hashing_enabled", True)),
            evidence_artifact_sha256=hashes,
            reproducibility_status=str(result.metadata.get("reproducibility_status", "not recorded")),
            scientific_limitations=[
                "Component attribution is software counterfactual attribution, not proof of external causal effects.",
                "Results depend on the supplied historical records.",
                "Replay does not prove production safety.",
                "Cohort summaries are descriptive.",
                "A Decision Contract pass is not proof of compliance, fairness, safety, or correctness.",
            ],
        )
        manifest["run_id"] = run_id
        from decishift.evidence import make_integrity_root
        manifest.pop("integrity_root", None)
        manifest["integrity_root"] = make_integrity_root(manifest)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return run_id

    def _path(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id or any(sep in run_id for sep in ("/", "\\")):
            raise FileNotFoundError(f"Unknown run: {run_id}")
        root = self.root.resolve()
        path = (root / run_id).resolve()
        if path.parent != root or not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        return path

    def report(self, run_id: str, format: str) -> str:
        path = self._path(run_id)
        mapping = {"terminal": "report.txt", "markdown": "report.md", "json": "summary.json", "html": "report.html"}
        if format not in mapping:
            raise ValueError("format must be terminal, markdown, json, or html")
        return (path / mapping[format]).read_text(encoding="utf-8")

    def report_path(self, run_id: str, format: str) -> Path:
        path = self._path(run_id)
        mapping = {"terminal": "report.txt", "markdown": "report.md", "json": "summary.json", "html": "report.html"}
        if format not in mapping:
            raise ValueError("format must be terminal, markdown, json, or html")
        return path / mapping[format]

    def verify(self, run_id: str):
        return verify_evidence_path(self._path(run_id))

    def explain(self, run_id: str, record_id: str) -> dict:
        path = self._path(run_id)
        records = pd.read_csv(path / "records.csv", dtype={"record_id": str})
        match = records[records["record_id"] == str(record_id)]
        if match.empty:
            raise KeyError(f"Record id not found: {record_id}")
        if len(match) > 1:
            raise KeyError(f"Record id is not unique in saved evidence: {record_id}")
        payload = match.iloc[0].to_dict()
        attr = pd.read_csv(path / "attribution.csv", dtype={"record_id": str})
        if not attr.empty:
            payload["component_attribution"] = attr[attr["record_id"] == str(record_id)].to_dict(orient="records")
        else:
            payload["component_attribution"] = "insufficient evidence"
        return payload

    def load(self, run_id: str) -> ComparisonResult:
        path = self._path(run_id)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            records = pd.read_csv(path / "records.csv")
            result = ComparisonResult(records=records, metadata={"evidence_schema_version": "0.1", "reproducibility_status": "not recorded in evidence schema v0.1"})
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema = str(manifest.get("evidence_schema_version", "0.1"))
            if schema not in {"1.0", "0.1"}:
                raise ValueError(f"Unsupported future evidence schema version: {schema}")
            records = pd.read_csv(path / "records.csv")
            metadata = {
                "evidence_schema_version": schema,
                "reproducibility_status": manifest.get("reproducibility_status", "not recorded"),
                "input_data_fingerprint": manifest.get("input_data_fingerprint"),
                "baseline_component_identities": manifest.get("baseline_component_identities", {}),
                "candidate_component_identities": manifest.get("candidate_component_identities", {}),
                "attribution_method": manifest.get("attribution_method"),
                "attribution_parameters": manifest.get("attribution_parameters", {}),
                "random_seed": manifest.get("random_seed"),
            }
            result = ComparisonResult(records=records, changed_components=list(manifest.get("changed_components", [])), metadata=metadata)
        for attr_name, filename in (("attribution", "attribution.csv"), ("interactions", "interactions.csv"), ("cohorts", "cohorts.csv")):
            fp = path / filename
            if fp.exists():
                try:
                    table = pd.read_csv(fp)
                except pd.errors.EmptyDataError:
                    table = pd.DataFrame()
                setattr(result, attr_name, table)
        fragility_path = path / "fragility.csv"
        if fragility_path.exists():
            fragility_records = pd.read_csv(fragility_path)
            summary = {}
            try:
                saved = json.loads((path / "summary.json").read_text(encoding="utf-8"))
                summary = saved.get("fragility") or {}
            except Exception:
                pass
            result.fragility = {"summary": summary, "records": fragility_records}
        outcome_path = path / "outcome.json"
        if outcome_path.exists():
            result.outcome_analysis = json.loads(outcome_path.read_text(encoding="utf-8"))
        try:
            saved = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            diag = saved.get("attribution_diagnostics")
            if diag:
                result.diagnostics = AttributionDiagnostics(**diag)
        except Exception:
            pass
        return result
