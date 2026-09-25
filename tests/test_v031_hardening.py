import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decishift.config import (
    _component,
    _identity_from_spec,
    _import_object,
    build_flow,
    load_data,
    load_yaml,
    resolve_data_path,
)
from decishift.core.exceptions import ConfigurationError
from decishift.core.pipeline import DecisionPipeline
from decishift.evidence import (
    canonical_json_bytes,
    fingerprint_dataframe,
    make_integrity_root,
    reproducibility_from_identities,
    sha256_file,
    verify_evidence_path,
)
from decishift.flow import DecisionFlow, DecisionNode
from decishift.flow.executor import FlowExecutor
from decishift.replay.engine import HybridReplayCache


class ReadingModel:
    version = "reading_v1"

    def __init__(self):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        if isinstance(features, pd.DataFrame):
            return features["x"].to_numpy(dtype=float)
        return np.asarray(features, dtype=float)[:, 0]


class PassFeatures:
    version = "features_v1"

    def transform(self, records):
        return records[["x"]].copy()


class MutatingModel:
    version = "mutating_model_v1"

    def predict(self, features):
        original = features["x"].to_numpy(dtype=float).copy()
        features.loc[:, "x"] = 0.0
        return original


def test_pipeline_shared_cache_is_bound_to_input_content():
    model = ReadingModel()
    pipeline = DecisionPipeline(model=model, threshold=0.5)
    cache = {}

    first = pipeline.evaluate(pd.DataFrame({"x": [0.1, 0.9]}), cache=cache)
    second = pipeline.evaluate(pd.DataFrame({"x": [0.7, 0.8]}), cache=cache)

    assert model.calls == 2
    assert first.raw_scores.tolist() == pytest.approx([0.1, 0.9])
    assert second.raw_scores.tolist() == pytest.approx([0.7, 0.8])
    fingerprints = {key[1] for key in cache if key and key[0] == "pipeline"}
    assert len(fingerprints) == 2


def test_pipeline_cached_intermediates_are_not_poisoned_by_mutating_components():
    records = pd.DataFrame({"x": [0.2, 0.8]})
    features = PassFeatures()
    baseline = DecisionPipeline(features=features, model=MutatingModel(), threshold=0.5)
    candidate = DecisionPipeline(features=features, model=ReadingModel(), threshold=0.5)
    replay = HybridReplayCache(baseline, candidate, records)

    baseline_trace = replay.evaluate(set())
    candidate_trace = replay.evaluate({"model"})

    assert baseline_trace.raw_scores.tolist() == pytest.approx([0.2, 0.8])
    assert candidate_trace.raw_scores.tolist() == pytest.approx([0.2, 0.8])


def test_pipeline_components_cannot_mutate_caller_owned_records():
    class MutatingFeatures:
        version = "features_mutating_v1"

        def transform(self, records):
            records.loc[:, "x"] = 0.25
            records["temporary"] = 1
            return records[["x"]]

    def mutating_threshold(records, scores):
        records.loc[:, "x"] = 99.0
        return np.full(len(records), 0.5)

    mutating_threshold.version = "threshold_mutating_v1"

    class MutatingRules:
        version = "rules_mutating_v1"

        def apply(self, records, scores, decisions):
            records.loc[:, "x"] = -1.0
            return decisions

    original = pd.DataFrame({"x": [0.1, 0.9]})
    expected = original.copy(deep=True)
    pipeline = DecisionPipeline(
        features=MutatingFeatures(),
        model=ReadingModel(),
        threshold=mutating_threshold,
        rules=MutatingRules(),
    )

    trace = pipeline.evaluate(original)

    pd.testing.assert_frame_equal(original, expected)
    assert trace.raw_scores.tolist() == pytest.approx([0.25, 0.25])


def test_hybrid_replay_cache_recomputes_after_external_record_mutation():
    records = pd.DataFrame({"x": [0.2, 0.8]})
    model = ReadingModel()
    pipeline = DecisionPipeline(model=model, threshold=0.5)
    replay = HybridReplayCache(pipeline, pipeline, records)

    first = replay.evaluate(set())
    records.loc[:, "x"] = [0.6, 0.7]
    second = replay.evaluate(set())

    assert model.calls == 2
    assert replay.evaluations == 2
    assert first.raw_scores.tolist() == pytest.approx([0.2, 0.8])
    assert second.raw_scores.tolist() == pytest.approx([0.6, 0.7])


def test_flow_nodes_receive_isolated_record_snapshots():
    class Mutator:
        version = "mutator_v1"

        def run(self, records, inputs):
            records.loc[:, "x"] = 999
            return pd.Series(["mutated"] * len(records), index=records.index)

    class Observer:
        version = "observer_v1"

        def run(self, records, inputs):
            return pd.Series(records["x"].to_numpy(), index=records.index)

    class Final:
        version = "final_v1"

        def run(self, records, inputs):
            return pd.Series(inputs["b_observer"].to_numpy(), index=records.index)

    records = pd.DataFrame({"x": [1, 2]})
    expected = records.copy(deep=True)
    flow = DecisionFlow(
        [
            DecisionNode("a_mutator", Mutator(), version="mutator_v1"),
            DecisionNode("b_observer", Observer(), version="observer_v1"),
            DecisionNode("final", Final(), ("a_mutator", "b_observer"), version="final_v1"),
        ],
        "final",
    )

    trace = FlowExecutor(records).evaluate(flow)

    pd.testing.assert_frame_equal(records, expected)
    assert trace.output("b_observer").tolist() == [1, 2]
    assert trace.actions.tolist() == [1, 2]


def test_evidence_helpers_cover_canonical_and_reproducibility_edges():
    payload = {
        "int": np.int64(3),
        "float": np.float32(1.5),
        "nan": np.float32(np.nan),
        "bool": np.bool_(True),
        "path": Path("artifact.bin"),
    }
    decoded = json.loads(canonical_json_bytes(payload))
    assert decoded == {"bool": True, "float": 1.5, "int": 3, "nan": None, "path": "artifact.bin"}
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json_bytes({"bad": object()})

    assert reproducibility_from_identities({}, {}) == "reproducible"
    assert reproducibility_from_identities({"a": {"reproducible": True}}, {}) == "reproducible"
    assert reproducibility_from_identities(
        {"a": {"reproducible": True}}, {"b": {"reproducible": False}}
    ) == "partially_reproducible"
    assert reproducibility_from_identities({"a": {"reproducible": False}}, {}) == "unstable"

    frame = pd.DataFrame({"obj": [{"b": 2, "a": 1}, {"a": 2}]})
    assert fingerprint_dataframe(frame) == fingerprint_dataframe(frame.copy(deep=True))
    assert fingerprint_dataframe(frame, include_content=False).startswith("sha256:")


def _write_manifest(run, artifact_map=None, **updates):
    manifest = {
        "evidence_schema_version": "1.0",
        "evidence_artifact_sha256": artifact_map or {},
    }
    manifest.update(updates)
    manifest["integrity_root"] = make_integrity_root(manifest)
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_evidence_verifier_failure_modes_are_explicit(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    assert verify_evidence_path(missing).message == "manifest.json is missing"

    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / "manifest.json").write_text("{", encoding="utf-8")
    assert "unreadable" in verify_evidence_path(unreadable).message

    future = tmp_path / "future"
    future.mkdir()
    (future / "manifest.json").write_text(
        json.dumps({"evidence_schema_version": "99.0", "integrity_root": "sha256:bad"}),
        encoding="utf-8",
    )
    assert "unsupported evidence schema" in verify_evidence_path(future).message

    noncanonical = tmp_path / "noncanonical"
    noncanonical.mkdir()
    (noncanonical / "manifest.json").write_text(
        '{"evidence_schema_version":"1.0","evidence_artifact_sha256":{},"bad":NaN,"integrity_root":"sha256:x"}',
        encoding="utf-8",
    )
    assert "cannot be canonicalized" in verify_evidence_path(noncanonical).message

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    good = artifacts / "good.txt"
    good.write_text("good", encoding="utf-8")
    _write_manifest(
        artifacts,
        {
            "good.txt": sha256_file(good),
            "missing.txt": "sha256:" + "0" * 64,
            "../outside.txt": "sha256:" + "0" * 64,
        },
    )
    result = verify_evidence_path(artifacts)
    statuses = {item.artifact: item.status for item in result.items}
    assert not result.passed
    assert statuses == {"../outside.txt": "FAIL", "good.txt": "PASS", "missing.txt": "MISSING"}


def explode_factory():
    raise RuntimeError("factory exploded")


def test_configuration_loader_rejects_unsafe_or_malformed_shapes(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="root must be a mapping"):
        load_yaml(bad_yaml)

    with pytest.raises(ConfigurationError, match="data.path is required"):
        resolve_data_path({}, base_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="does not exist"):
        resolve_data_path({"path": "missing.csv"}, base_dir=tmp_path)

    unsupported = tmp_path / "data.txt"
    unsupported.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Only local CSV and Parquet"):
        load_data({"path": unsupported.name}, base_dir=tmp_path)

    with pytest.raises(ConfigurationError, match="module:object"):
        _import_object("json")
    with pytest.raises(ConfigurationError, match="Cannot import"):
        _import_object("json:not_present")
    with pytest.raises(ConfigurationError, match="Factory .* failed"):
        _component("model", {"factory": "tests.test_v031_hardening:explode_factory"})

    with pytest.raises(ConfigurationError, match="non-empty mapping"):
        build_flow({}, "flow")
    with pytest.raises(ConfigurationError, match="final_node is required"):
        build_flow({"nodes": {"a": {"object": "json:dumps"}}}, "flow")
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        build_flow({"nodes": {"a": "bad"}, "final_node": "a"}, "flow")
    with pytest.raises(ConfigurationError, match="depends_on must be a list"):
        build_flow(
            {"nodes": {"a": {"object": "json:dumps", "depends_on": "bad"}}, "final_node": "a"},
            "flow",
        )


def test_configuration_artifact_identity_and_threshold_value(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    identity, artifact_path = _identity_from_spec(
        "model",
        {"version": "m1", "artifact": artifact.name},
        base_dir=tmp_path,
    )
    assert identity is not None
    assert identity.source == "artifact"
    assert identity.digest == sha256_file(artifact)
    assert artifact_path == str(artifact.resolve())

    explicit, _ = _identity_from_spec("model", {"digest": "abcd"})
    assert explicit is not None and explicit.digest == "sha256:abcd"

    value, version, threshold_identity, threshold_artifact = _component(
        "threshold",
        {"value": "0.42", "version": "policy_v1"},
    )
    assert value == pytest.approx(0.42)
    assert version == "policy_v1"
    assert threshold_identity is not None
    assert threshold_artifact is None

    with pytest.raises(ConfigurationError, match="Artifact path"):
        _identity_from_spec("model", {"artifact": "missing.bin"}, base_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="require 'object' or 'factory'"):
        _component("model", {"version": "m1"})
