import numpy as np
import pandas as pd
import pytest

from decishift.config import build_pipeline
from decishift.core.exceptions import ComponentExecutionError, ReproducibilityError
from decishift.core.identity import sha256_file
from decishift.core.pipeline import DecisionPipeline
from decishift.diff.compare import compare_pipelines
from tests.helpers import pipeline


def test_explicit_component_versions_are_reproducible():
    p = pipeline()
    assert p.reproducibility_status == "reproducible"
    assert all(i.reproducible for i in p.component_identities().values())


def test_artifact_sha256_identity(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model artifact")
    class Model:
        def predict(self, x): return np.asarray(x)[:,0]
    p = DecisionPipeline(model=Model(), threshold=0.5, artifact_paths={"model": artifact}, versions={"features":"identity","calibrator":"identity","threshold":"t","rules":"identity"})
    ident = p.component_identity("model")
    assert ident.digest == sha256_file(artifact)
    assert ident.reproducible


def test_unstable_component_is_marked_and_strict_mode_rejects():
    class Stateful:
        def __init__(self): self.weight = 1.0
        def predict(self, x): return np.asarray(x)[:,0] * self.weight
    data = pd.DataFrame({"x":[0.1,0.9]})
    loose = DecisionPipeline(model=Stateful(), threshold=0.5)
    assert loose.component_identity("model").source == "unstable"
    strict = DecisionPipeline(model=Stateful(), threshold=0.5, strict_reproducibility=True)
    with pytest.raises(ReproducibilityError): strict.evaluate(data)


def test_duplicate_and_null_ids_rejected():
    b = pipeline(); c = pipeline(threshold=0.4, labels={"threshold":"t2"})
    with pytest.raises(ValueError, match="unique"):
        compare_pipelines(b,c,pd.DataFrame({"id":[1,1],"x":[0.2,0.8],"force":False}),id_column="id")
    with pytest.raises(ValueError, match="null"):
        compare_pipelines(b,c,pd.DataFrame({"id":[1,None],"x":[0.2,0.8],"force":False}),id_column="id")


def test_reordered_feature_index_rejected():
    class Reorder:
        version="r1"
        def transform(self, frame): return frame[["x"]].iloc[::-1]
    class Model:
        version="m1"
        def predict(self, x): return np.asarray(x)[:,0]
    p=DecisionPipeline(features=Reorder(),model=Model(),threshold=0.5,versions={"features":"r1","model":"m1","threshold":"t","calibrator":"identity","rules":"identity"})
    with pytest.raises(ValueError, match="reordered"):
        p.evaluate(pd.DataFrame({"x":[0.2,0.8]}, index=[10,20]))


def test_nan_score_inf_threshold_wrong_length_and_internal_typeerror_surface():
    data=pd.DataFrame({"x":[0.2,0.8]})
    class NaNModel:
        version="m"
        def predict(self,x): return np.array([np.nan,0.8])
    with pytest.raises(ValueError, match="NaN or infinite"):
        DecisionPipeline(model=NaNModel(),versions={"model":"m"}).evaluate(data)
    class ShortModel:
        version="m"
        def predict(self,x): return np.array([0.2])
    with pytest.raises(ValueError, match="one value per record"):
        DecisionPipeline(model=ShortModel(),versions={"model":"m"}).evaluate(data)
    class Model:
        version="m"
        def predict(self,x): return np.array([0.2,0.8])
    with pytest.raises(ValueError, match="NaN or infinite"):
        DecisionPipeline(model=Model(),threshold=float("inf"),versions={"model":"m","threshold":"bad"}).evaluate(data)
    def broken_threshold(records, scores):
        raise TypeError("unsupported operand inside threshold")
    broken_threshold.version="threshold_v3"
    with pytest.raises(ComponentExecutionError, match="TypeError: unsupported operand inside threshold"):
        DecisionPipeline(model=Model(),threshold=broken_threshold,versions={"model":"m","threshold":"threshold_v3"}).evaluate(data)


def dynamic_threshold(records, scores):
    return np.where(records["region"].to_numpy()=="north",0.4,0.6)
dynamic_threshold.version="regional_v3"


def test_dynamic_threshold_yaml_object():
    p = build_pipeline({"model":{"factory":"tests.test_v02_reproducibility_pipeline:IdentityModel","version":"m1"},"threshold":{"object":"tests.test_v02_reproducibility_pipeline:dynamic_threshold","version":"regional_v3"}},"p")
    data=pd.DataFrame({"x":[0.5,0.5],"region":["north","south"]})
    trace=p.evaluate(data)
    assert trace.decisions.tolist()==[1,0]


class IdentityModel:
    version="m1"
    def predict(self,x): return np.asarray(x["x"] if isinstance(x,pd.DataFrame) else x) if not isinstance(x,pd.DataFrame) else x["x"].to_numpy()


def test_closure_source_identity_is_not_overclaimed():
    scale = 2.0
    def closure_model(x):
        return np.asarray(x)[:, 0] * scale
    p = DecisionPipeline(model=closure_model, threshold=0.5)
    ident = p.component_identity("model")
    assert ident.source == "unstable"
    assert not ident.reproducible


def test_unstable_same_class_instances_still_compare_within_process():
    class Stateful:
        def __init__(self, weight): self.weight = weight
        def predict(self, x): return np.asarray(x)[:, 0] * self.weight
    data = pd.DataFrame({"x": [0.4, 0.6]})
    baseline = DecisionPipeline(model=Stateful(1.0), threshold=0.5)
    candidate = DecisionPipeline(model=Stateful(2.0), threshold=0.5)
    assert "model" in baseline.changed_components(candidate)
    result = compare_pipelines(baseline, candidate, data)
    assert result.summary()["changed_decisions"] == 1
    assert result.metadata["reproducibility_status"] == "partially_reproducible"
