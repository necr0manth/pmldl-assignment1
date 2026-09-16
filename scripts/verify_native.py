"""Verify two real native DVC/MLflow training runs; Docker is NOT mocked as passed.

Used by Windows CI where no Docker Desktop Linux engine is installed.
The Linux CI job separately verifies full Docker deployment and browsers.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from fastapi.testclient import TestClient
from pmldl.api import create_app
from pmldl.common import write_json
from pmldl.example import verify


def main():
    evidence = {"status": "running", "platform": platform.platform(),
                "python": sys.version, "commit": os.environ.get("GITHUB_SHA"),
                "docker_desktop_tested": False, "runs": []}
    try:
        subprocess.run([sys.executable, "pipeline.py", "--help"], cwd=ROOT, check=True)
        evidence["example"] = verify(ROOT, ROOT / "models/example")
        for _ in range(2):
            subprocess.run([sys.executable, "pipeline.py", "run", "--train-only"], cwd=ROOT, check=True)
            record = json.loads((ROOT / "runs/latest.json").read_text(encoding="utf-8"))
            metadata = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
            assert record["status"] == "succeeded" and record["target"] == "train"
            assert record["return_code"] == 0
            # Reload the actual model produced by DVC, then exercise FastAPI.
            with TestClient(create_app(ROOT / "models/model.joblib")) as client:
                response = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5,
                                                        "petal_length": 1.4, "petal_width": 0.2})
                assert response.status_code == 200
                result = response.json()
                assert result["species"] == "setosa" and result["model_run_id"] == metadata["run_id"]
            # Export verification also reproduces the partitions/metrics on this OS.
            verify(ROOT, ROOT / "models")
            evidence["runs"].append({**record, "model_run_id": metadata["run_id"],
                                     "prediction": result, "metrics": metadata["metrics"]})
        assert evidence["runs"][0]["model_run_id"] != evidence["runs"][1]["model_run_id"]
        subprocess.run([sys.executable, "pipeline.py", "status"], cwd=ROOT, check=True)
        evidence["status"] = "passed"
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(ROOT / "reports/native/summary.json", evidence)


if __name__ == "__main__":
    main()
