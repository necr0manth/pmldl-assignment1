"""Keep a small, verified model snapshot separate from generated DVC outputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from pmldl.common import CLASSES, FEATURES, ROOT, parameters, sha256, utc_now, write_json
from pmldl.datasets import prepare
from pmldl.scheduler import exclusive_lock



def text_hash_matches(path: Path, expected: str) -> bool:
    """Accept LF/CRLF-only differences in text, never relax model-byte hashes.

    Existing Windows checkouts can contain CRLF; pandas also writes the native
    line ending. Both representations describe the same input/source/CSV.
    Compare known byte digests, without ignoring any other content changes.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return expected in {
        hashlib.sha256(data).hexdigest(),
        hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest(),
    }


def verify(root: Path, directory: Path) -> dict:
    """Verify hashes before loading our trusted pickle; reproduce holdout metrics."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if sha256(directory / "model.joblib") != metadata["model_sha256"]:
        raise ValueError("Example model hash mismatch")
    for field, path in (("raw_sha256", root / "data/raw/iris.csv"),
                        ("params_sha256", root / "params.yaml")):
        if not text_hash_matches(path, metadata[field]):
            raise ValueError(f"Example is outdated: {path.name} changed; regenerate the snapshot")
    for name, digest in metadata["source_hashes"].items():
        if Path(name).name != name or not text_hash_matches(root / "code/pmldl" / name, digest):
            raise ValueError(f"Example training source changed: {name}")
    for name, version in metadata["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Install the pinned model requirements: expected {name}=={version}")
    bundle = joblib.load(directory / "model.joblib")
    if bundle["schema_version"] != 1 or bundle["metadata"]["run_id"] != metadata["run_id"]:
        raise ValueError("Example metadata does not match the model bundle")
    model = bundle["model"]
    if bundle["metadata"]["features"] != FEATURES or list(model.classes_) != CLASSES:
        raise ValueError("Example model schema mismatch")
    with tempfile.TemporaryDirectory() as name:
        output = Path(name)
        report = prepare(root / "data/raw/iris.csv", output, **parameters(root)["data"])
        for partition in ("train", "test"):
            if not text_hash_matches(output / f"{partition}.csv", metadata[f"{partition}_sha256"]):
                raise ValueError(f"Example {partition} partition no longer reproduces")
        test = pd.read_csv(output / "test.csv")
        predicted = model.predict(test[FEATURES])
        probabilities = model.predict_proba(test[FEATURES])
        observed = {
            "accuracy": float(accuracy_score(test.species, predicted)),
            "f1_macro": float(f1_score(test.species, predicted, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(test.species, predicted, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(test.species, predicted, average="macro", zero_division=0)),
            "log_loss": float(log_loss(test.species, probabilities, labels=model.classes_)),
        }
        if not all(np.isclose(value, metadata["metrics"][metric], rtol=0, atol=1e-10)
                   for metric, value in observed.items()):
            raise ValueError("Example holdout metrics do not match metadata")
    return {"status": "verified", "model_run_id": metadata["run_id"],
            "model_sha256": metadata["model_sha256"], "metrics": observed,
            "train_rows": report["train_rows"], "test_rows": report["test_rows"]}


def export(root: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}; choose an empty destination")
    # Verify the generated artifact before creating a candidate for Git review.
    result = verify(root, root / "models")
    destination.mkdir(parents=True)
    for name in ("model.joblib", "metadata.json"):
        shutil.copyfile(root / "models" / name, destination / name)
    write_json(destination / "provenance.json", {
        **result, "exported_at": utc_now(),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "purpose": "Small checked-in example; normal scheduled runs always train a new model.",
    })
    return result


def install(root: Path, directory: Path) -> dict:
    with exclusive_lock(root / "runs/.scheduler.lock"):
        with exclusive_lock(root / "runs/.pipeline.lock"):
            result = verify(root, directory)
            for name in ("model.joblib", "metadata.json"):
                target = root / "models" / name
                temporary = target.with_name(name + ".install-tmp")
                try:
                    shutil.copyfile(directory / name, temporary)
                    temporary.chmod(0o644)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "install", "export"))
    parser.add_argument("--directory", type=Path, default=ROOT / "models/example")
    args = parser.parse_args()
    action = {"verify": verify, "install": install, "export": export}[args.command]
    print(json.dumps(action(ROOT, args.directory.resolve()), indent=2))


if __name__ == "__main__":
    main()
