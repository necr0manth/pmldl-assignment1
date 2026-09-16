"""Real browser tests: same tabs across two 300-second scheduled Docker deployments.

No accelerated clocks, substituted API responses, or mocked Docker operations.
Use --existing to test already-running containers (including nondefault ports).
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "code"))
from pmldl.deployment import compose_command, service_url
from pmldl.scheduler import process_group_options, stop_process_tree

EVIDENCE = ROOT / "reports/browser"
INTERVAL = 300
LABELS = {"sepal_length": "Sepal length (cm)", "sepal_width": "Sepal width (cm)",
          "petal_length": "Petal length (cm)", "petal_width": "Petal width (cm)"}
CASES = [("setosa", [5.1, 3.5, 1.4, 0.2]), ("versicolor", [6.0, 2.9, 4.5, 1.5]),
         ("virginica", [6.5, 3.0, 5.8, 2.2])]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def http_json(url: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def compose(*args: str) -> None:
    subprocess.run([*compose_command(ROOT), *args], cwd=ROOT, check=True, timeout=300)


def wait_for_run(process, previous_ids: set[str], deadline: float, pages: list) -> dict:
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
            raise RuntimeError(f"Scheduler exited early: {process.returncode}")
        # Pump Playwright events while keeping the same browser tabs alive.
        if pages:
            pages[0].wait_for_timeout(1000)
        else:
            time.sleep(1)
    raise TimeoutError("Timed out waiting for the real scheduled pipeline")


def check_predictions(page, phase: str, metadata: dict, api: str) -> dict:
    run_id = metadata["run_id"]
    expect(page.get_by_role("heading", name="Iris classifier", exact=True)).to_be_visible(timeout=90000)
    expect(page.get_by_role("spinbutton")).to_have_count(4)
    button = page.get_by_role("button", name="Predict", exact=True)
    expect(button).to_be_enabled(timeout=90000)
    results = []
    for species, measurements in CASES:
        payload = dict(zip(LABELS, measurements, strict=True))
        reference = http_json(api + "/predict", payload)
        if reference["species"] != species or reference["model_run_id"] != run_id:
            raise AssertionError(f"Unexpected reference prediction: {reference}")
        for feature, label in LABELS.items():
            field = page.get_by_role("spinbutton", name=label, exact=True)
            field.fill(str(payload[feature]))
            field.press("Tab")
        button.click()
        expect(page.get_by_text(f"Predicted species: {species}", exact=True)).to_be_visible(timeout=30000)
        expect(page.get_by_text(f"Model run: {run_id}", exact=True)).to_be_visible(timeout=30000)
        expect(page.get_by_test_id("stProgress")).to_have_count(3)
        rendered = {}
        for name, probability in reference["probabilities"].items():
            label = f"{name}: {probability:.2%}"
            element = page.get_by_text(label, exact=True)
            expect(element).to_be_visible(timeout=15000)
            rendered[name] = element.inner_text()
        expect(page.get_by_test_id("stException")).to_have_count(0)
        results.append({"inputs": payload, **reference, "rendered_probabilities": rendered})
        page.screenshot(path=str(EVIDENCE / f"{phase}-{species}.png"), full_page=True)
    details = page.get_by_test_id("stExpander").locator("details")
    if not details.evaluate("element => element.open"):
        details.locator("summary").click()
    block = page.get_by_test_id("stJson")
    expect(block).to_be_visible()
    expect(block).to_contain_text(run_id)
    for metric in metadata["metrics"]:
        expect(block).to_contain_text(metric)
    page.screenshot(path=str(EVIDENCE / f"{phase}-metrics.png"), full_page=True)
    print(f"BROWSER_OK {phase}: three real submissions, probabilities, model ID and metrics", flush=True)
    return {"phase": phase, "model_run_id": run_id, "predictions": results, "metrics_panel": True}


def check_outage(pages: list, expected_run_id: str) -> dict:
    try:
        compose("stop", "api")
        for name, page in pages:
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_text(re.compile("Prediction failed\\."))).to_be_visible(timeout=30000)
            expect(page.get_by_text(re.compile("^Predicted species:"))).to_have_count(0)
            expect(page.get_by_test_id("stProgress")).to_have_count(0)
            page.screenshot(path=str(EVIDENCE / f"{name}-api-stopped.png"), full_page=True)
    finally:
        compose("up", "-d", "--wait", "--wait-timeout", "120", "api")
    for name, page in pages:
        page.get_by_role("button", name="Predict", exact=True).click()
        expect(page.get_by_text("Predicted species: virginica", exact=True)).to_be_visible(timeout=30000)
        expect(page.get_by_text(f"Model run: {expected_run_id}", exact=True)).to_be_visible()
        page.screenshot(path=str(EVIDENCE / f"{name}-api-restored.png"), full_page=True)
    return {"real_api_stop": True, "stale_results_removed": True, "recovered": True}


def check_mlflow(browser, metadata: dict) -> dict:
    from mlflow.tracking import MlflowClient
    uri = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(metadata["run_id"])
    process = None
    context = browser.new_context()
    page = context.new_page()
    try:
        with (EVIDENCE / "mlflow-server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri,
                                        "--host", "127.0.0.1", "--port", "5000"],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **process_group_options())
            deadline = time.monotonic() + 90
            while True:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.monotonic() > deadline or process.poll() is not None:
                        raise RuntimeError("MLflow UI did not start; inspect mlflow-server.log")
                    time.sleep(1)
            page.goto(f"http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
                      wait_until="domcontentloaded", timeout=60000)
            expect(page.get_by_text("accuracy", exact=True).first).to_be_visible(timeout=60000)
            expect(page.get_by_text("f1_macro", exact=True).first).to_be_visible(timeout=30000)
            expect(page.locator("body")).to_contain_text(run.info.run_name)
            page.screenshot(path=str(EVIDENCE / "mlflow-run.png"), full_page=True)
            return {"run_id": run.info.run_id, "run_name": run.info.run_name, "metrics_visible": True}
    except BaseException:
        page.screenshot(path=str(EVIDENCE / "mlflow-failure.png"), full_page=True)
        (EVIDENCE / "mlflow-failure.html").write_text(page.content(), encoding="utf-8")
        raise
    finally:
        context.close()
        if process is not None:
            stop_process_tree(process, timeout=20)


def main() -> None:
    global EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--browsers", nargs="+", choices=("chromium", "firefox"), default=["chromium", "firefox"])
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    EVIDENCE = args.evidence_dir.resolve()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    api, app = service_url("API_PORT", "8000"), service_url("APP_PORT", "8501")
    evidence = {"status": "running", "commit": os.environ.get("GITHUB_SHA"), "api": api, "app": app,
                "runs": [], "browser_checks": [], "interval_seconds": None if args.existing else INTERVAL}
    process = None
    handles = []
    previous = {path.stem for path in (ROOT / "runs").glob("*.json") if path.stem != "latest"}
    try:
        with (EVIDENCE / "scheduler.log").open("w", encoding="utf-8") as log, sync_playwright() as playwright:
            if not args.existing:
                process = subprocess.Popen([sys.executable, "pipeline.py", "schedule", "--interval", str(INTERVAL),
                                            "--max-runs", "2"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           **process_group_options())
            try:
                for name in args.browsers:
                    browser = getattr(playwright, name).launch(headless=True)
                    context = browser.new_context(viewport={"width": 1280, "height": 1050})
                    context.tracing.start(screenshots=True, snapshots=True, sources=True)
                    page = context.new_page()
                    errors = []
                    page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                    handles.append((name, browser, context, page, errors))
                for phase in (("existing",) if args.existing else ("first", "second")):
                    if process is not None:
                        record = wait_for_run(process, previous, time.monotonic() + 1200,
                                              [h[3] for h in handles])
                        previous.add(record["run_id"])
                    metadata = read_json(ROOT / "models/metadata.json")
                    if http_json(api + "/health")["model_run_id"] != metadata["run_id"]:
                        raise AssertionError("API serves a stale model")
                    if process is not None:
                        receipt = read_json(ROOT / "reports/deployment.json")
                        if receipt["status"] != "deployed" or receipt["model_run_id"] != metadata["run_id"]:
                            raise AssertionError("Deployment receipt is stale")
                        evidence["runs"].append({**record, "model_run_id": metadata["run_id"]})
                        for name, value in (("metadata", metadata), ("deployment", receipt)):
                            (EVIDENCE / f"{phase}-{name}.json").write_text(json.dumps(value, indent=2))
                    for name, browser, context, page, errors in handles:
                        if phase != "second":
                            page.goto(app, wait_until="domcontentloaded", timeout=60000)
                        # No navigation/reload/replacement on second: old tabs must reconnect.
                        check = check_predictions(page, f"{name}-{phase}", metadata, api)
                        if errors:
                            raise AssertionError(f"{name} JavaScript errors: {errors}")
                        evidence["browser_checks"].append({**check, "browser_version": browser.version,
                                                          "same_tab_after_redeploy": phase == "second",
                                                          "javascript_errors": errors.copy()})
                if process is not None:
                    process.wait(timeout=30)
                    if process.returncode:
                        raise RuntimeError(f"Scheduler exited with {process.returncode}")
                    first, second = evidence["runs"]
                    if first["model_run_id"] == second["model_run_id"]:
                        raise AssertionError("Model was not retrained")
                    gap = (datetime.fromisoformat(second["started_at"]) - datetime.fromisoformat(first["started_at"])).total_seconds()
                    if first["duration_seconds"] >= INTERVAL or not 295 <= gap <= 330:
                        raise AssertionError(f"Expected 300-second start spacing, got {gap}")
                    evidence["observed_start_gap_seconds"] = gap
                    evidence["api_failure_check"] = check_outage([(h[0], h[3]) for h in handles], metadata["run_id"])
                    evidence["mlflow_ui"] = check_mlflow(handles[0][1], metadata)
                evidence["status"] = "passed"
            except BaseException:
                for name, browser, context, page, errors in handles:
                    page.screenshot(path=str(EVIDENCE / f"{name}-failure.png"), full_page=True)
                    (EVIDENCE / f"{name}-failure.html").write_text(page.content(), encoding="utf-8")
                raise
            finally:
                for name, browser, context, page, errors in handles:
                    context.tracing.stop(path=str(EVIDENCE / f"{name}-trace.zip"))
                    context.close()
                    browser.close()
    except BaseException as error:
        evidence.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                stop_process_tree(process, timeout=10)
        (EVIDENCE / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
