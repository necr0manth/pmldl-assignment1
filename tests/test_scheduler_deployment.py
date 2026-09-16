import json
import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from pmldl import deployment, scheduler
from pmldl.common import sha256, write_json


@pytest.mark.parametrize("finished,expected", [(40, 300), (299, 300), (301, 600), (650, 900)])
def test_scheduler_skips_missed_slots(finished, expected):
    assert scheduler.next_slot(0, finished, 300) == expected


def test_scheduled_command_forces_complete_graph():
    command = scheduler.repro_command()
    assert "--force" in command and "--no-run-cache" in command
    assert command[-1] == "deploy"
    assert scheduler.repro_command(True)[-1] == "train"


def test_lock_released_after_error(tmp_path):
    path = tmp_path / "pipeline.lock"
    with scheduler.exclusive_lock(path):
        with pytest.raises(scheduler.AlreadyRunning):
            with scheduler.exclusive_lock(path):
                pass
    with pytest.raises(RuntimeError):
        with scheduler.exclusive_lock(path):
            raise RuntimeError("failed")
    with scheduler.exclusive_lock(path):
        pass


@pytest.mark.parametrize("code,status", [(0, "succeeded"), (3, "failed")])
def test_runner_preserves_exit_status_and_logs(tmp_path, monkeypatch, code, status):
    monkeypatch.setattr(scheduler, "repro_command",
                        lambda train_only: [sys.executable, "-c", f"print('stage output'); raise SystemExit({code})"])
    assert scheduler.run_once(tmp_path, train_only=True) == code
    record = json.loads((tmp_path / "runs/latest.json").read_text())
    assert record["status"] == status and record["return_code"] == code
    assert "stage output" in (tmp_path / record["log_file"]).read_text()


def test_missing_docker_is_reported(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("docker not installed")
    monkeypatch.setattr(scheduler.subprocess, "run", unavailable)
    assert scheduler.run_once(tmp_path) == 1
    record = json.loads((tmp_path / "runs/latest.json").read_text())
    assert record["status"] == "failed" and "docker" in record["error"]


def test_scheduler_retries_failure_without_waiting_in_test(tmp_path, monkeypatch):
    run = Mock(side_effect=[1, 0])
    sleep = Mock()
    monkeypatch.setattr(scheduler, "run_once", run)
    monkeypatch.setattr(scheduler.time, "sleep", sleep)
    scheduler.schedule(tmp_path, max_runs=2)
    assert run.call_count == 2 and sleep.call_count == 1
    assert 0 < sleep.call_args.args[0] <= 300


@pytest.mark.parametrize("kwargs", [{"interval": 299}, {"max_runs": 0}])
def test_invalid_schedule_fails(tmp_path, kwargs):
    with pytest.raises(ValueError):
        scheduler.schedule(tmp_path, **kwargs)


def deployment_metadata(artifact):
    root = artifact.parent.parent
    write_json(root / "models/metadata.json", {"run_id": "test-run", "model_sha256": sha256(artifact)})
    return root


def test_deployment_builds_before_recreate_and_checks_new_model(artifact, monkeypatch):
    root = deployment_metadata(artifact)
    run = Mock()
    monkeypatch.setattr(deployment.subprocess, "run", run)
    monkeypatch.setattr(deployment, "request_json", Mock(side_effect=[
        {"model_run_id": "test-run"},
        {"model_run_id": "test-run", "species": "setosa", "probabilities": {"setosa": 1.0}},
    ]))
    monkeypatch.setattr(deployment.urllib.request, "urlopen", lambda *a, **kw: nullcontext(Mock(status=200)))
    report = deployment.deploy(root)
    assert report["status"] == "deployed"
    assert run.call_args_list[0].args[0][-3:] == ["build", "api", "app"]
    assert "--force-recreate" in run.call_args_list[1].args[0]
    assert "--wait" in run.call_args_list[1].args[0]
    assert (root / "reports/deployment.json").is_file()


def test_failed_build_does_not_recreate_running_services(artifact, monkeypatch):
    root = deployment_metadata(artifact)
    run = Mock(side_effect=subprocess.CalledProcessError(1, "docker build"))
    monkeypatch.setattr(deployment.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        deployment.deploy(root)
    assert run.call_count == 1
    assert not (root / "reports/deployment.json").exists()


def test_deployment_detects_stale_model(artifact, monkeypatch):
    root = deployment_metadata(artifact)
    monkeypatch.setattr(deployment.subprocess, "run", Mock())
    monkeypatch.setattr(deployment, "request_json", Mock(return_value={"model_run_id": "old-run"}))
    with pytest.raises(RuntimeError, match="stale"):
        deployment.deploy(root)


def test_deployment_rejects_mismatched_artifact(artifact):
    root = deployment_metadata(artifact)
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="does not match"):
        deployment.deploy(root)
