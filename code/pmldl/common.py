"""Shared input schema and artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASSES = ["setosa", "versicolor", "virginica"]
TARGET = "species"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomic replacement: readers never observe partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def parameters(root: Path = ROOT) -> dict:
    import yaml
    with (root / "params.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
