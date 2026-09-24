from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVIDENCE_SCHEMA_VERSION = "1.0"
SUPPORTED_EVIDENCE_SCHEMA_VERSIONS = {"1.0", "0.1"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if not np.isfinite(val):
            return None
        return val
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def fingerprint_dataframe(frame: pd.DataFrame, *, include_content: bool = True) -> str:
    schema = [
        {"name": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]
    prefix = canonical_json_bytes({"schema": schema, "index_dtype": str(frame.index.dtype), "rows": len(frame)})
    if not include_content:
        return sha256_bytes(prefix + b"|content-hashing-disabled")
    # pandas' stable row hashing avoids materializing a large textual CSV while
    # incorporating values, schema, index and row order. Object columns can
    # contain otherwise-unhashable values, so fall back to a deterministic
    # canonical representation rather than silently skipping content.
    try:
        row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype="uint64")
        content = row_hashes.tobytes(order="C")
    except TypeError:
        normalized = frame.copy()
        for column in normalized.columns:
            if normalized[column].dtype == object:
                normalized[column] = normalized[column].map(
                    lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
                )
        content = normalized.to_csv(index=True, lineterminator="\n", na_rep="<NA>").encode("utf-8")
    return sha256_bytes(prefix + b"|" + content)


def reproducibility_from_identities(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    values = []
    for mapping in (baseline, candidate):
        for value in mapping.values():
            if isinstance(value, dict):
                values.append(bool(value.get("reproducible", False)))
    if not values or all(values):
        return "reproducible"
    if any(values):
        return "partially_reproducible"
    return "unstable"


def make_integrity_root(manifest_without_root: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(manifest_without_root))


def build_manifest(
    *,
    decishift_version: str,
    attribution_method: str | None,
    attribution_parameters: dict[str, Any] | None,
    random_seed: int | None,
    number_of_records: int,
    changed_components: list[str],
    baseline_component_identities: dict[str, Any] | None,
    candidate_component_identities: dict[str, Any] | None,
    configuration_sha256: str | None,
    input_data_fingerprint: str | None,
    input_hashing_enabled: bool,
    evidence_artifact_sha256: dict[str, str],
    reproducibility_status: str,
    scientific_limitations: list[str],
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "decishift_version": decishift_version,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "attribution_method": attribution_method,
        "attribution_parameters": attribution_parameters or {},
        "random_seed": random_seed,
        "number_of_records": int(number_of_records),
        "changed_components": list(changed_components),
        "baseline_component_identities": baseline_component_identities or {},
        "candidate_component_identities": candidate_component_identities or {},
        "configuration_sha256": configuration_sha256,
        "input_data_fingerprint": input_data_fingerprint,
        "input_content_hashing_enabled": bool(input_hashing_enabled),
        "evidence_artifact_sha256": dict(sorted(evidence_artifact_sha256.items())),
        "reproducibility_status": reproducibility_status,
        "scientific_limitations": list(scientific_limitations),
    }
    manifest["integrity_root"] = make_integrity_root(manifest)
    return manifest


@dataclass
class VerificationItem:
    artifact: str
    expected: str | None
    actual: str | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    run_id: str
    passed: bool
    integrity_root: str | None
    expected_integrity_root: str | None
    items: list[VerificationItem]
    manifest_status: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "integrity_root": self.integrity_root,
            "expected_integrity_root": self.expected_integrity_root,
            "manifest_status": self.manifest_status,
            "items": [item.as_dict() for item in self.items],
            "message": self.message,
        }


def verify_evidence_path(path: str | Path) -> VerificationResult:
    run_path = Path(path)
    manifest_path = run_path / "manifest.json"
    run_id = run_path.name
    if not manifest_path.exists():
        return VerificationResult(
            run_id, False, None, None, [], "FAIL", "manifest.json is missing"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return VerificationResult(run_id, False, None, None, [], "FAIL", f"manifest.json is unreadable: {exc}")

    expected_root = manifest.get("integrity_root")
    root_input = dict(manifest)
    root_input.pop("integrity_root", None)
    try:
        actual_root = make_integrity_root(root_input)
    except (TypeError, ValueError) as exc:
        return VerificationResult(run_id, False, None, expected_root, [], "FAIL", f"manifest cannot be canonicalized: {exc}")
    manifest_ok = bool(expected_root and expected_root == actual_root)

    items: list[VerificationItem] = []
    all_artifacts_ok = True
    for artifact, expected in sorted((manifest.get("evidence_artifact_sha256") or {}).items()):
        artifact_path = run_path / artifact
        # Prevent a crafted manifest from traversing outside the evidence bundle.
        try:
            resolved = artifact_path.resolve()
            root = run_path.resolve()
        except OSError:
            resolved = artifact_path
            root = run_path
        if root != resolved.parent and root not in resolved.parents:
            items.append(VerificationItem(artifact, expected, None, "FAIL"))
            all_artifacts_ok = False
            continue
        if not artifact_path.exists() or not artifact_path.is_file():
            items.append(VerificationItem(artifact, expected, None, "MISSING"))
            all_artifacts_ok = False
            continue
        actual = sha256_file(artifact_path)
        ok = actual == expected
        items.append(VerificationItem(artifact, expected, actual, "PASS" if ok else "FAIL"))
        all_artifacts_ok = all_artifacts_ok and ok

    passed = manifest_ok and all_artifacts_ok
    return VerificationResult(
        run_id=run_id,
        passed=passed,
        integrity_root=actual_root,
        expected_integrity_root=expected_root,
        items=items,
        manifest_status="PASS" if manifest_ok else "FAIL",
        message="Evidence bundle verified." if passed else "Evidence verification FAILED.",
    )
