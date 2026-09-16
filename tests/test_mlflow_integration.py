import json

import pytest

mlflow = pytest.importorskip("mlflow", reason="Install requirements-dev.txt for the MLflow integration test")

from pmldl.common import FEATURES, sha256
from pmldl.training import train_and_package


def test_training_logs_real_model_and_metrics(project, partitions):
    metadata = train_and_package(project)
    assert metadata["model_sha256"] == sha256(project / "models/model.joblib")
    run = mlflow.tracking.MlflowClient().get_run(metadata["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["accuracy"] == metadata["metrics"]["accuracy"]
    loaded = mlflow.sklearn.load_model(metadata["mlflow_model_uri"])
    assert len(loaded.predict(partitions[1][FEATURES])) == len(partitions[1])
    assert json.loads((project / "reports/metrics.json").read_text()) == metadata["metrics"]
