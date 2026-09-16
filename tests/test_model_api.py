import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pmldl.api import create_app
from pmldl.common import FEATURES
from pmldl.deployment import SAMPLE
from pmldl.training import fit_evaluate, save_bundle


def test_features_metrics_and_roundtrip(trained, partitions, tmp_path):
    model, metrics, report = trained
    assert report["feature_count"] == 14
    assert 0.9 <= metrics["accuracy"] <= 1
    assert 0.9 <= metrics["f1_macro"] <= 1
    assert metrics["log_loss"] >= 0
    assert sum(map(sum, report["confusion_matrix"])) == len(partitions[1])
    path = tmp_path / "model.joblib"
    save_bundle(path, model, {"test": True})
    loaded = joblib.load(path)["model"]
    np.testing.assert_array_equal(model.predict(partitions[1][FEATURES]),
                                  loaded.predict(partitions[1][FEATURES]))
    assert path.stat().st_mode & 0o444 == 0o444


def test_test_features_do_not_influence_scaling(partitions):
    model, _, _ = fit_evaluate(*partitions)
    different_test = partitions[1].copy()
    different_test[FEATURES] *= 100
    again, _, _ = fit_evaluate(partitions[0], different_test)
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, again.named_steps["scale"].mean_)
    np.testing.assert_array_equal(model.named_steps["classifier"].coef_, again.named_steps["classifier"].coef_)


def test_api_serves_real_predictions(artifact):
    with TestClient(create_app(artifact)) as client:
        assert client.get("/health").json() == {"status": "ok", "model_run_id": "test-run"}
        assert client.get("/model").json()["features"] == FEATURES
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 200
        prediction = response.json()
        assert prediction["species"] == "setosa"
        assert prediction["model_run_id"] == "test-run"
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
        assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("bad", [0, -1, 31, "five", None, True, float("nan"), float("inf")])
def test_api_rejects_invalid_measurements(artifact, bad):
    with TestClient(create_app(artifact)) as client:
        payload = dict(SAMPLE, sepal_length=bad)
        response = client.post("/predict", content=json.dumps(payload),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["detail"]


@pytest.mark.parametrize("kind", ["missing", "extra", "malformed"])
def test_api_rejects_invalid_schema(artifact, kind):
    payload = SAMPLE.copy()
    if kind == "missing":
        del payload["sepal_length"]
    elif kind == "extra":
        payload["unexpected"] = 42
    with TestClient(create_app(artifact)) as client:
        if kind == "malformed":
            response = client.post("/predict", content="{bad json", headers={"Content-Type": "application/json"})
        else:
            response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_api_missing_model_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Model not found"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass


def test_api_rejects_incompatible_artifact(artifact):
    bundle = joblib.load(artifact)
    bundle["schema_version"] = 999
    joblib.dump(bundle, artifact)
    with pytest.raises(RuntimeError, match="Incompatible"):
        with TestClient(create_app(artifact)):
            pass
