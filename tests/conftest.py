import shutil

import pandas as pd
import pytest

from pmldl.common import CLASSES, FEATURES, ROOT
from pmldl.datasets import prepare
from pmldl.training import fit_evaluate, save_bundle


@pytest.fixture
def project(tmp_path):
    for relative in ["data/raw/iris.csv", "params.yaml", "requirements/model.txt"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    return tmp_path


@pytest.fixture
def partitions(project):
    prepare(project / "data/raw/iris.csv", project / "data/processed")
    return (pd.read_csv(project / "data/processed/train.csv"),
            pd.read_csv(project / "data/processed/test.csv"))


@pytest.fixture
def trained(partitions):
    return fit_evaluate(*partitions)


@pytest.fixture
def artifact(project, trained):
    model, metrics, _ = trained
    path = project / "models/model.joblib"
    save_bundle(path, model, {"run_id": "test-run", "features": FEATURES,
                              "classes": CLASSES, "trained_at": "2026-01-01T00:00:00+00:00",
                              "metrics": metrics})
    return path
