import importlib.metadata
import json
import shutil

import pytest
from pmldl.common import ROOT, FEATURES, CLASSES, sha256, write_json
from pmldl.example import verify, install
from pmldl.scheduler import exclusive_lock, AlreadyRunning
from pmldl.training import save_bundle


@pytest.fixture
def example(project, trained, partitions):
    directory = project / "models/example"
    directory.mkdir(parents=True)
    sources = {}
    for name in ("datasets.py", "training.py"):
        target = project / "code/pmldl" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "code/pmldl" / name, target)
        sources[name] = sha256(target)
    model, metrics, _ = trained
    metadata = {"run_id": "fixture", "features": FEATURES, "classes": CLASSES, "metrics": metrics,
                "raw_sha256": sha256(project / "data/raw/iris.csv"),
                "params_sha256": sha256(project / "params.yaml"), "source_hashes": sources,
                "versions": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")}}
    for part in ("train", "test"):
        metadata[f"{part}_sha256"] = sha256(project / "data/processed" / f"{part}.csv")
    save_bundle(directory / "model.joblib", model, metadata)
    metadata["model_sha256"] = sha256(directory / "model.joblib")
    write_json(directory / "metadata.json", metadata)
    return directory


def test_example_metrics_reproduce(project, example):
    assert verify(project, example)["status"] == "verified"


def test_example_install_preserves_snapshot(project, example):
    before = sha256(example / "model.joblib")
    install(project, example)
    assert sha256(project / "models/model.joblib") == before == sha256(example / "model.joblib")


@pytest.mark.parametrize("name", ["model", "params", "source"])
def test_example_detects_changes_before_serving(project, example, name):
    path = {"model": example / "model.joblib", "params": project / "params.yaml",
            "source": project / "code/pmldl/datasets.py"}[name]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        verify(project, example)


def test_example_install_does_not_race_scheduler(project, example):
    with exclusive_lock(project / "runs/.scheduler.lock"):
        with pytest.raises(AlreadyRunning):
            install(project, example)
