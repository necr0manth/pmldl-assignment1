"""Real Chromium -> Streamlit -> FastAPI test over two five-minute scheduled runs.

Run in a fresh checkout after installing requirements-dev.txt,
requirements-browser.txt and `python -m playwright install --with-deps chromium`.
Docker must be running. No time, network or model mocks are used.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/browser"
INTERVAL = 300
LABELS = {
    "sepal_length": "Sepal length (cm)",
    "sepal_width": "Sepal width (cm)",
    "petal_length": "Petal length (cm)",
    "petal_width": "Petal width (cm)",
}
CASES = [
    ("setosa", [5.1, 3.5, 1.4, 0.2]),
    ("versicolor", [6.0, 2.9, 4.5, 1.5]),
    ("virginica", [6.5, 3.0, 5.8, 2.2]),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def http_json(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def compose(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["docker", "compose", "-p", "pmldl-iris", "-f",
         str(ROOT / "code/deployment/docker-compose.yml"), *args],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout or ""


def wait_for_run(process: subprocess.Popen, previous_ids: set[str], deadline: float) -> dict:
    while time.monotonic() < deadline:
        latest = ROOT / "runs/latest.json"
        if latest.exists():
            record = read_json(latest)
            if record["run_id"] not in previous_ids:
                if record["status"] in {"failed", "cancelled"}:
                    raise RuntimeError(f"Scheduled pipeline failed: {record}")
                if record["status"] == "succeeded":
                    return record
        if process.poll() is not None:
            raise RuntimeError(f"Scheduler exited before completing the run: {process.returncode}")
        time.sleep(1)
    raise TimeoutError("Timed out waiting for a real scheduled pipeline run")


def verify_browser(browser, phase: str, expected_run_id: str) -> dict:
    context = browser.new_context(viewport={"width": 1280, "height": 1050})
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    results = []
    try:
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded", timeout=60000)
        expect(page.get_by_role("heading", name="Iris classifier", exact=True)).to_be_visible(timeout=30000)
        expect(page.get_by_role("spinbutton")).to_have_count(4)
        expect(page.get_by_role("button", name="Predict", exact=True)).to_be_visible()
        for species, measurements in CASES:
            payload = dict(zip(LABELS, measurements, strict=True))
            reference = http_json("http://127.0.0.1:8000/predict", payload)
            if reference["species"] != species or reference["model_run_id"] != expected_run_id:
                raise AssertionError(f"Unexpected API prediction: {reference}")
            for feature, label in LABELS.items():
                field = page.get_by_role("spinbutton", name=label, exact=True)
                field.fill(str(payload[feature]))
                field.press("Tab")
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_text(f"Predicted species: {species}", exact=True)).to_be_visible(timeout=30000)
            expect(page.get_by_text(f"Model run: {expected_run_id}", exact=True)).to_be_visible(timeout=30000)
            expect(page.get_by_test_id("stVegaLiteChart")).to_be_visible(timeout=15000)
            expect(page.get_by_test_id("stException")).to_have_count(0)
            page.screenshot(path=str(EVIDENCE / f"{phase}-{species}.png"), full_page=True)
            results.append({"inputs": payload, "species": species,
                            "model_run_id": expected_run_id, "api_probabilities": reference["probabilities"]})
        if page_errors:
            raise AssertionError(f"Browser JavaScript errors: {page_errors}")
        print(f"BROWSER_OK {phase}: three form submissions and probability charts; model={expected_run_id}", flush=True)
        return {"phase": phase, "model_run_id": expected_run_id, "predictions": results,
                "browser_version": browser.version, "javascript_errors": page_errors}
    except BaseException:
        page.screenshot(path=str(EVIDENCE / f"{phase}-failure.png"), full_page=True)
        (EVIDENCE / f"{phase}-failure.html").write_text(page.content(), encoding="utf-8")
        raise
    finally:
        context.tracing.stop(path=str(EVIDENCE / f"{phase}-trace.zip"))
        context.close()


def verify_api_failure(browser) -> dict:
    """Prove the running frontend depends on the real API, not a mocked response."""
    context = browser.new_context(viewport={"width": 1280, "height": 1050})
    page = context.new_page()
    try:
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        button = page.get_by_role("button", name="Predict", exact=True)
        expect(button).to_be_visible(timeout=30000)
        button.click()
        expect(page.get_by_text("Predicted species: setosa", exact=True)).to_be_visible(timeout=30000)
        compose("stop", "api")
        button.click()
        expect(page.get_by_text(re.compile("Prediction failed\\."))).to_be_visible(timeout=30000)
        expect(page.get_by_text(re.compile("^Predicted species:"))).to_have_count(0)
        page.screenshot(path=str(EVIDENCE / "api-stopped-error.png"), full_page=True)
        compose("up", "-d", "--wait", "--wait-timeout", "120", "api")
        button.click()
        expect(page.get_by_text("Predicted species: setosa", exact=True)).to_be_visible(timeout=30000)
        page.screenshot(path=str(EVIDENCE / "api-restored.png"), full_page=True)
        print("BROWSER_OK real API stopped: error displayed, stale prediction removed; recovery succeeded", flush=True)
        return {"real_api_stop": True, "error_displayed": True, "stale_prediction_removed": True, "recovered": True}
    except BaseException:
        page.screenshot(path=str(EVIDENCE / "api-failure-test-failed.png"), full_page=True)
        raise
    finally:
        context.close()


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    # Isolate this smoke test from optional user overrides; the actual pipeline is unchanged.
    for name in ("BIND_ADDRESS", "API_PORT", "APP_PORT", "COMPOSE_PROJECT_NAME"):
        if name in os.environ:
            raise RuntimeError(f"Unset {name} before running this localhost CI smoke test")
    previous_ids = {path.stem for path in (ROOT / "runs").glob("*.json") if path.stem != "latest"}
    evidence = {"status": "running", "interval_seconds": INTERVAL,
                "commit": os.environ.get("GITHUB_SHA"), "runs": [], "browser_checks": []}
    process = None
    try:
        with (EVIDENCE / "scheduler.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "pipeline.py", "schedule", "--interval", str(INTERVAL), "--max-runs", "2"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 1200
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    for phase in ("first", "second"):
                        record = wait_for_run(process, previous_ids, deadline)
                        previous_ids.add(record["run_id"])
                        metadata = read_json(ROOT / "models/metadata.json")
                        receipt = read_json(ROOT / "reports/deployment.json")
                        if receipt["status"] != "deployed" or receipt["model_run_id"] != metadata["run_id"]:
                            raise AssertionError("Deployment receipt does not match the newly trained model")
                        health = http_json("http://127.0.0.1:8000/health")
                        if health["model_run_id"] != metadata["run_id"]:
                            raise AssertionError("API health exposes a stale model")
                        record["model_run_id"] = metadata["run_id"]
                        evidence["runs"].append(record)
                        # Preserve both receipts: DVC overwrites the latest report on the next run.
                        for name, value in (("metadata", metadata), ("deployment", receipt)):
                            (EVIDENCE / f"{phase}-{name}.json").write_text(json.dumps(value, indent=2) + "\n")
                        print(f"SCHEDULE_OK {phase}: {record['started_at']} -> {record['finished_at']}; model={metadata['run_id']}", flush=True)
                        evidence["browser_checks"].append(verify_browser(browser, phase, metadata["run_id"]))
                    process.wait(timeout=30)
                    if process.returncode != 0:
                        raise RuntimeError(f"Scheduler exited with {process.returncode}")
                    first, second = evidence["runs"]
                    if first["model_run_id"] == second["model_run_id"]:
                        raise AssertionError("The second automatic run did not create a new model")
                    gap = (datetime.fromisoformat(second["started_at"]) - datetime.fromisoformat(first["started_at"])).total_seconds()
                    if first["duration_seconds"] >= INTERVAL or not 295 <= gap <= 330:
                        raise AssertionError(f"Expected two real 300-second slots, observed start gap {gap}")
                    evidence["observed_start_gap_seconds"] = gap
                    evidence["api_failure_check"] = verify_api_failure(browser)
                    evidence["status"] = "passed"
                    print(f"E2E_PASSED: two scheduled Docker deployments; real start gap={gap:.3f}s; six browser predictions; API failure/recovery checked", flush=True)
                finally:
                    browser.close()
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        (EVIDENCE / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
