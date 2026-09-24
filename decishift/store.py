from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from decishift.attribution.diagnostics import AttributionDiagnostics
from decishift.diff.compare import ComparisonResult
from decishift.evidence import build_flow_manifest, build_manifest, make_integrity_root, sha256_file, verify_evidence_path
from decishift.flow.attribution import FlowAttributionDiagnostics
from decishift.flow.compare import FlowComparisonResult
from decishift.flow.reports import render_flow_html, render_flow_json, render_flow_markdown, render_flow_terminal
from decishift.reports.render import render_html, render_json, render_markdown, render_terminal
from decishift.version import __version__

_EMPTY_ATTR = ["record_id", "component", "score_contribution", "decision_contribution"]
_EMPTY_INTERACTIONS = ["record_id", "component_a", "component_b", "score_interaction", "decision_interaction", "interaction_only_flip"]
_EMPTY_COHORTS = ["dimension", "cohort", "size", "coverage", "flip_rate"]
_EMPTY_FLOW_ATTR = ["record_id", "player", "attribution_target", "target_contribution"]
_EMPTY_FLOW_INTERACTIONS = [
    "record_id", "player_a", "player_b", "attribution_target", "target_interaction",
    "interaction_only_transition", "resulting_transition",
]
_EMPTY_FLOW_COHORTS = [
    "dimension", "cohort", "size", "coverage", "action_shift_rate",
    "global_action_shift_rate", "excess_action_shift_rate", "most_common_transition",
]


@dataclass
class RunStore:
    root: Path = Path(".decishift/runs")

    def save(self, result: ComparisonResult | FlowComparisonResult) -> str:
        if getattr(result, "metadata", {}).get("mode") == "flow":
            return self._save_flow(result)
        return self._save_pipeline(result)

    def _new_run_path(self) -> tuple[str, Path]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=False)
        return run_id, path

    def _write_common_artifacts(
        self,
        path: Path,
        result,
        *,
        empty_attr: list[str],
        empty_interactions: list[str],
        empty_cohorts: list[str],
        flow: bool,
    ) -> list[str]:
        result.records.to_csv(path / "records.csv", index=False)
        (result.attribution if result.attribution is not None else pd.DataFrame(columns=empty_attr)).to_csv(
            path / "attribution.csv", index=False
        )
        (result.interactions if result.interactions is not None else pd.DataFrame(columns=empty_interactions)).to_csv(
            path / "interactions.csv", index=False
        )
        (result.cohorts if result.cohorts is not None else pd.DataFrame(columns=empty_cohorts)).to_csv(
            path / "cohorts.csv", index=False
        )
        if result.fragility is not None and isinstance(result.fragility.get("records"), pd.DataFrame):
            result.fragility["records"].to_csv(path / "fragility.csv", index=False)
        if result.outcome_analysis is not None:
            (path / "outcome.json").write_text(
                json.dumps(result.outcome_analysis, indent=2, allow_nan=False, default=str), encoding="utf-8"
            )

        if flow:
            (path / "summary.json").write_text(render_flow_json(result), encoding="utf-8")
            (path / "report.md").write_text(render_flow_markdown(result), encoding="utf-8")
            (path / "report.txt").write_text(render_flow_terminal(result), encoding="utf-8")
            (path / "report.html").write_text(render_flow_html(result), encoding="utf-8")
            topology = {
                "baseline": result.metadata.get("baseline_topology"),
                "candidate": result.metadata.get("candidate_topology"),
                "diff": result.metadata.get("topology_diff"),
            }
            (path / "topology.json").write_text(
                json.dumps(topology, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
        else:
            (path / "summary.json").write_text(render_json(result), encoding="utf-8")
            (path / "report.md").write_text(render_markdown(result), encoding="utf-8")
            (path / "report.txt").write_text(render_terminal(result), encoding="utf-8")
            (path / "report.html").write_text(render_html(result), encoding="utf-8")

        artifacts = [
            "summary.json", "records.csv", "attribution.csv", "interactions.csv",
            "cohorts.csv", "report.md", "report.txt", "report.html",
        ]
        if flow:
            artifacts.append("topology.json")
        for optional in ("fragility.csv", "outcome.json"):
            if (path / optional).exists():
                artifacts.append(optional)
        return artifacts

    def _save_pipeline(self, result: ComparisonResult) -> str:
        run_id, path = self._new_run_path()
        artifacts = self._write_common_artifacts(
            path, result, empty_attr=_EMPTY_ATTR, empty_interactions=_EMPTY_INTERACTIONS,
            empty_cohorts=_EMPTY_COHORTS, flow=False,
        )
        hashes = {name: sha256_file(path / name) for name in sorted(artifacts)}
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
        manifest.pop("integrity_root", None)
        manifest["integrity_root"] = make_integrity_root(manifest)
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        return run_id

    def _save_flow(self, result: FlowComparisonResult) -> str:
        run_id, path = self._new_run_path()
        artifacts = self._write_common_artifacts(
            path, result, empty_attr=_EMPTY_FLOW_ATTR, empty_interactions=_EMPTY_FLOW_INTERACTIONS,
            empty_cohorts=_EMPTY_FLOW_COHORTS, flow=True,
        )
        hashes = {name: sha256_file(path / name) for name in sorted(artifacts)}
        summary = result.summary()
        diagnostics = (
            result.diagnostics.as_dict() if result.diagnostics is not None and hasattr(result.diagnostics, "as_dict")
            else result.diagnostics or {}
        )
        manifest = build_flow_manifest(
            decishift_version=__version__,
            number_of_records=len(result.records),
            configuration_sha256=result.metadata.get("configuration_sha256"),
            input_data_fingerprint=result.metadata.get("input_data_fingerprint"),
            input_hashing_enabled=bool(result.metadata.get("input_content_hashing_enabled", True)),
            evidence_artifact_sha256=hashes,
            reproducibility_status=str(result.metadata.get("reproducibility_status", "not recorded")),
            baseline_topology=result.metadata.get("baseline_topology") or {},
            candidate_topology=result.metadata.get("candidate_topology") or {},
            baseline_topology_digest=str(result.metadata.get("baseline_topology_digest", "")),
            candidate_topology_digest=str(result.metadata.get("candidate_topology_digest", "")),
            topology_compatible=bool(result.metadata.get("topology_compatible", False)),
            topology_diff=result.metadata.get("topology_diff") or {},
            baseline_node_identities=result.metadata.get("baseline_node_identities") or {},
            candidate_node_identities=result.metadata.get("candidate_node_identities") or {},
            changed_nodes=list(result.changed_nodes),
            attribution_groups=result.metadata.get("attribution_groups") or {},
            final_node=str(result.metadata.get("final_node", "")),
            transition_matrix=summary.get("transition_matrix") or {},
            structural_impact=result.metadata.get("structural_impact") or {},
            attribution_status=str(result.metadata.get("attribution_status", "not requested")),
            attribution_target=result.metadata.get("attribution_target"),
            attribution_method=result.metadata.get("attribution_method"),
            attribution_parameters=result.metadata.get("attribution_parameters") or {},
            sampling_diagnostics=diagnostics,
            random_seed=result.metadata.get("random_seed"),
            scientific_limitations=[
                "Structural reachability is not causal impact and does not imply observed historical behavior changed.",
                "Node/group attribution is software counterfactual attribution, not proof of external causal effects.",
                "Categorical actions are never numerically subtracted.",
                "Sampling intervals quantify permutation-sampling uncertainty only.",
                "Topology-changing node attribution is unsupported in DeciShift v0.3.",
                "Results depend on the supplied historical records.",
                "A Decision Contract pass is not proof of compliance, fairness, safety, or correctness.",
            ],
        )
        manifest["run_id"] = run_id
        manifest.pop("integrity_root", None)
        manifest["integrity_root"] = make_integrity_root(manifest)
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
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
        manifest = {}
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        is_flow = str(manifest.get("evidence_schema_version")) == "2.0"
        if not attr.empty:
            rows = attr[attr["record_id"] == str(record_id)].to_dict(orient="records")
            payload["node_attribution" if is_flow else "component_attribution"] = rows
        else:
            payload["node_attribution" if is_flow else "component_attribution"] = "insufficient evidence"
        return payload

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def load(self, run_id: str) -> ComparisonResult | FlowComparisonResult:
        path = self._path(run_id)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            records = pd.read_csv(path / "records.csv")
            result: ComparisonResult | FlowComparisonResult = ComparisonResult(
                records=records,
                metadata={"evidence_schema_version": "0.1", "reproducibility_status": "not recorded in evidence schema v0.1"},
            )
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema = str(manifest.get("evidence_schema_version", "0.1"))
            if schema not in {"2.0", "1.0", "0.1"}:
                raise ValueError(f"Unsupported future evidence schema version: {schema}")
            records = pd.read_csv(path / "records.csv")
            if schema == "2.0":
                metadata = {
                    "mode": "flow",
                    "evidence_schema_version": schema,
                    "reproducibility_status": manifest.get("reproducibility_status", "not recorded"),
                    "input_data_fingerprint": manifest.get("input_data_fingerprint"),
                    "input_content_hashing_enabled": manifest.get("input_content_hashing_enabled", True),
                    "baseline_topology": manifest.get("baseline_topology", {}),
                    "candidate_topology": manifest.get("candidate_topology", {}),
                    "baseline_topology_digest": manifest.get("baseline_topology_digest"),
                    "candidate_topology_digest": manifest.get("candidate_topology_digest"),
                    "topology_compatible": manifest.get("topology_compatible", False),
                    "topology_diff": manifest.get("topology_diff", {}),
                    "baseline_node_identities": manifest.get("baseline_node_identities", {}),
                    "candidate_node_identities": manifest.get("candidate_node_identities", {}),
                    "attribution_groups": manifest.get("attribution_groups", {}),
                    "final_node": manifest.get("final_node"),
                    "structural_impact": manifest.get("structural_impact", {}),
                    "attribution_status": manifest.get("attribution_status"),
                    "attribution_target": manifest.get("attribution_target"),
                    "attribution_method": manifest.get("attribution_method"),
                    "attribution_parameters": manifest.get("attribution_parameters", {}),
                    "random_seed": manifest.get("random_seed"),
                }
                result = FlowComparisonResult(
                    records=records,
                    changed_nodes=list(manifest.get("changed_nodes", [])),
                    metadata=metadata,
                )
            else:
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
                result = ComparisonResult(
                    records=records,
                    changed_components=list(manifest.get("changed_components", [])),
                    metadata=metadata,
                )

        for attr_name, filename in (("attribution", "attribution.csv"), ("interactions", "interactions.csv"), ("cohorts", "cohorts.csv")):
            fp = path / filename
            if fp.exists():
                setattr(result, attr_name, self._read_table(fp))

        fragility_path = path / "fragility.csv"
        if fragility_path.exists():
            fragility_records = self._read_table(fragility_path)
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
                if isinstance(result, FlowComparisonResult):
                    diag = dict(diag)
                    diag.pop("converged", None)
                    result.diagnostics = FlowAttributionDiagnostics(**diag)
                else:
                    result.diagnostics = AttributionDiagnostics(**diag)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
        return result
