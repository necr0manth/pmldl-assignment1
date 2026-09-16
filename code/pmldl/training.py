"""Stage 2: engineer features, train, evaluate, track in MLflow, and package."""
from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pmldl.common import CLASSES, FEATURES, ROOT, TARGET, parameters, sha256, utc_now, write_json


def fit_evaluate(train: pd.DataFrame, test: pd.DataFrame, *, degree=2, C=1.0,
                 max_iter=1000, random_state=42):
    model = Pipeline([
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)),
    ])
    model.fit(train[FEATURES], train[TARGET])
    predicted = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])
    metrics = {
        "accuracy": float(accuracy_score(test[TARGET], predicted)),
        "f1_macro": float(f1_score(test[TARGET], predicted, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(test[TARGET], predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test[TARGET], predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(test[TARGET], probabilities, labels=model.classes_)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Model produced nonfinite evaluation metrics")
    report = {
        "classes": CLASSES, "train_rows": len(train), "test_rows": len(test),
        "feature_count": int(model.named_steps["polynomial"].n_output_features_),
        "confusion_matrix": confusion_matrix(test[TARGET], predicted, labels=CLASSES).tolist(),
        "classification_report": classification_report(
            test[TARGET], predicted, labels=CLASSES, output_dict=True, zero_division=0
        ),
    }
    return model, metrics, report


def save_bundle(path: Path, model: Pipeline, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump({"schema_version": 1, "model": model, "metadata": metadata}, name)
        os.chmod(name, 0o644)  # Readable by the non-root API container user.
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def train_and_package(root: Path = ROOT) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    root = root.resolve()
    train = pd.read_csv(root / "data/processed/train.csv")
    test = pd.read_csv(root / "data/processed/test.csv")
    config = parameters(root)["model"]
    model, metrics, evaluation = fit_evaluate(train, test, **config)
    # File-backed SQLite tracking does not require a running MLflow server.
    tracking_uri = "sqlite:///" + (root / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "iris-classification"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        experiment_name, artifact_location=(root / "mlartifacts").as_uri()
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name="iris-" + utc_now()) as run:
        mlflow.log_params(config)
        mlflow.log_params({"train_rows": len(train), "test_rows": len(test)})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(evaluation, "evaluation.json")
        mlflow.log_artifact(str(root / "data/processed/cleaning_report.json"))
        requirements = (root / "requirements/model.txt").read_text().splitlines()
        logged = mlflow.sklearn.log_model(
            sk_model=model, name="iris-classifier",
            signature=infer_signature(train[FEATURES], model.predict(train[FEATURES])),
            input_example=train[FEATURES].head(2), pip_requirements=requirements,
        )
        metadata = {
            "run_id": run.info.run_id, "trained_at": utc_now(),
            "features": FEATURES, "classes": list(model.classes_),
            "metrics": metrics, "mlflow_model_uri": logged.model_uri,
            "train_sha256": sha256(root / "data/processed/train.csv"),
            "test_sha256": sha256(root / "data/processed/test.csv"),
            "versions": {name: importlib.metadata.version(name) for name in
                         ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]},
        }
        # Log everything before promoting the packaged model for deployment.
        mlflow.log_dict(metadata, "metadata.json")
        save_bundle(root / "models/model.joblib", model, metadata)
        metadata["model_sha256"] = sha256(root / "models/model.joblib")
        write_json(root / "models/metadata.json", metadata)
        write_json(root / "reports/metrics.json", metrics)
        write_json(root / "reports/evaluation.json", evaluation)
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_and_package(), indent=2))
