"""Stage 3: build both images, recreate services, and verify a real prediction."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path

from pmldl.common import ROOT, sha256, utc_now, write_json

SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5,
          "petal_length": 1.4, "petal_width": 0.2}


def compose_command(root: Path = ROOT) -> list[str]:
    return ["docker", "compose", "--project-name",
            os.environ.get("COMPOSE_PROJECT_NAME", "pmldl-iris"),
            "--file", str(root / "code/deployment/docker-compose.yml")]


def service_url(port_variable: str, default_port: str) -> str:
    host = os.environ.get("BIND_ADDRESS", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1" if host == "0.0.0.0" else "[::1]"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{os.environ.get(port_variable, default_port)}"


def request_json(url: str, data: dict | None = None) -> dict:
    request = urllib.request.Request(
        url, data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def deploy(root: Path = ROOT) -> dict:
    metadata = json.loads((root / "models/metadata.json").read_text())
    if sha256(root / "models/model.joblib") != metadata["model_sha256"]:
        raise RuntimeError("Model file does not match its metadata; retrain before deploying")
    command = compose_command(root)
    # Build first: a failed build leaves the previous containers running.
    # Recreating containers is a simple demo rollout, not zero-downtime deployment.
    subprocess.run([*command, "build", "api", "app"], cwd=root, check=True)
    subprocess.run([*command, "up", "--detach", "--force-recreate", "--wait",
                    "--wait-timeout", "180", "api", "app"], cwd=root, check=True)
    api = service_url("API_PORT", "8000")
    app = service_url("APP_PORT", "8501")
    health = request_json(api + "/health")
    prediction = request_json(api + "/predict", SAMPLE)
    if health["model_run_id"] != metadata["run_id"] or prediction["model_run_id"] != metadata["run_id"]:
        raise RuntimeError("API is serving a stale model")
    if prediction["species"] != "setosa" or abs(sum(prediction["probabilities"].values()) - 1) > 1e-6:
        raise RuntimeError("Prediction smoke test failed")
    with urllib.request.urlopen(app + "/_stcore/health", timeout=15) as response:
        if response.status != 200:
            raise RuntimeError("Streamlit health check failed")
    report = {"status": "deployed", "deployed_at": utc_now(),
              "model_run_id": metadata["run_id"], "model_sha256": metadata["model_sha256"],
              "api_url": api, "app_url": app, "smoke_prediction": prediction}
    write_json(root / "reports/deployment.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(deploy(), indent=2))
