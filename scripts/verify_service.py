"""Exercise a real systemd user service on a disposable Linux CI runner.

Starts a full deployment, kills the scheduler, verifies automatic restart and a
second successful deployment, stops the service and checks both process locks.
Does not claim to test a host reboot or the user's own login session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from pmldl.common import write_json
from pmldl.scheduler import exclusive_lock
from pmldl.service import install

UNIT = "pmldl-ci-verification.service"


def ctl(*args, check=True):
    return subprocess.run(["systemctl", "--user", *args], check=check,
                          text=True, capture_output=True, timeout=45)


def wait_success(previous: set[str]) -> dict:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        path = ROOT / "runs/latest.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["run_id"] not in previous:
                if record["status"] == "failed":
                    raise RuntimeError(f"Service pipeline failed: {record}")
                if record["status"] == "succeeded":
                    return record
        time.sleep(1)
    raise TimeoutError("Systemd did not complete a new successful pipeline run")


def main() -> None:
    report = {"status": "running", "host_reboot_tested": False, "logout_tested": False}
    directory = Path.home() / ".config/systemd/user"
    target = directory / UNIT
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite an existing unit: {target}")
    output = ROOT / "reports/systemd"
    output.mkdir(parents=True, exist_ok=True)
    previous = {p.stem for p in (ROOT / "runs").glob("*.json") if p.stem != "latest"}
    try:
        target = install(ROOT, Path(sys.executable), directory, UNIT)
        (output / UNIT).write_text(target.read_text())
        subprocess.run(["systemd-analyze", "--user", "verify", str(target)], check=True, timeout=30)
        ctl("daemon-reload")
        ctl("enable", "--now", UNIT)
        assert ctl("is-enabled", UNIT).stdout.strip() == "enabled"
        first = wait_success(previous)
        previous.add(first["run_id"])
        old_pid = ctl("show", UNIT, "--property=MainPID", "--value").stdout.strip()
        ctl("kill", "--kill-whom=main", "--signal=SIGKILL", UNIT)
        second = wait_success(previous)
        new_pid = ctl("show", UNIT, "--property=MainPID", "--value").stdout.strip()
        restarts = int(ctl("show", UNIT, "--property=NRestarts", "--value").stdout.strip())
        assert new_pid != old_pid and int(new_pid) > 0 and restarts >= 1
        ctl("stop", UNIT)
        assert ctl("is-active", UNIT, check=False).returncode != 0
        with exclusive_lock(ROOT / "runs/.scheduler.lock"):
            with exclusive_lock(ROOT / "runs/.pipeline.lock"):
                pass
        report.update(status="passed", first_run=first, restarted_run=second,
                      old_pid=old_pid, new_pid=new_pid, automatic_restarts=restarts,
                      enabled=True, stopped=True, locks_released=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        logs = subprocess.run(["journalctl", "--user", "-u", UNIT, "--no-pager"],
                              text=True, capture_output=True, timeout=30)
        (output / "journal.log").write_text(logs.stdout + logs.stderr)
        ctl("disable", "--now", UNIT, check=False)
        target.unlink(missing_ok=True)
        ctl("daemon-reload", check=False)
        write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
