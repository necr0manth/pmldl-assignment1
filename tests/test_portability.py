"""Real OS locks/process trees; these tests also run on native Windows CI."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from pmldl import scheduler
from pmldl.common import ROOT
from pmldl.example import text_hash_matches


def subprocess_env():
    return {**os.environ, "PYTHONPATH": str(ROOT / "code"),
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def wait_for_record(path, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            if process.poll() is not None:
                raise AssertionError(f"Child exited before readiness: {process.returncode}")
            time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


def alive(pid):
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def assert_stopped(*pids):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.05)
    assert not any(alive(pid) for pid in pids), f"Process tree still alive: {pids}"


def test_exclusive_lock_blocks_another_process_and_recovers_after_kill(tmp_path):
    lock = tmp_path / "pipeline.lock"
    ready = tmp_path / "ready.json"
    code = ("import json, os, time; from pathlib import Path; "
            "from pmldl.scheduler import exclusive_lock\n"
            f"with exclusive_lock(Path({str(lock)!r})):\n"
            f"    Path({str(ready)!r}).write_text(json.dumps({{'pid': os.getpid()}}))\n"
            "    time.sleep(60)\n")
    process = subprocess.Popen([sys.executable, "-c", code], env=subprocess_env(),
                               **scheduler.process_group_options())
    try:
        holder = wait_for_record(ready, process)
        with pytest.raises(scheduler.AlreadyRunning):
            with scheduler.exclusive_lock(lock):
                pass
        psutil.Process(holder["pid"]).kill()
        process.wait(timeout=10)
        # A leftover pathname is not a held lock, even after a hard kill.
        with scheduler.exclusive_lock(lock):
            pass
    finally:
        scheduler.stop_process_tree(process, timeout=10)


def make_silent_tree(tmp_path):
    path = tmp_path / "проект with spaces"
    path.mkdir()
    ready = path / "tree.json"
    worker = path / "worker.py"
    worker.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).write_text(json.dumps({'parent': os.getpid(), 'child': child.pid}))\n"
        "time.sleep(60)\n", encoding="utf-8")
    return worker, ready


def test_stop_process_tree_kills_descendants_not_unrelated_processes(tmp_path):
    worker, ready = make_silent_tree(tmp_path)
    process = subprocess.Popen([sys.executable, str(worker), str(ready)],
                               **scheduler.process_group_options())
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 **scheduler.process_group_options())
    try:
        pids = wait_for_record(ready, process)
        scheduler.stop_process_tree(process, timeout=5)
        assert_stopped(pids["parent"], pids["child"])
        assert bystander.poll() is None
    finally:
        scheduler.stop_process_tree(process, timeout=10)
        scheduler.stop_process_tree(bystander, timeout=10)


def test_cli_interrupts_silent_child_tree_and_releases_lock(tmp_path):
    worker, ready = make_silent_tree(tmp_path)
    # Only the expensive DVC command is replaced here. The entry point,
    # interrupt signal, subprocesses and native locks are all real.
    code = ("import sys; from pathlib import Path; from pmldl import scheduler\n"
            "original = scheduler.run_once\n"
            f"scheduler.run_once = lambda **kw: original(Path({str(tmp_path)!r}), **kw)\n"
            f"scheduler.repro_command = lambda _: [sys.executable, {str(worker)!r}, {str(ready)!r}]\n"
            "raise SystemExit(scheduler.main(['run', '--train-only']))\n")
    with (tmp_path / "controller.log").open("w", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, "-c", code], env=subprocess_env(),
                                   stdout=output, stderr=subprocess.STDOUT,
                                   **scheduler.process_group_options())
        try:
            pids = wait_for_record(ready, process)
            process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
            assert process.wait(timeout=15) == 130
            assert_stopped(pids["parent"], pids["child"])
            record = json.loads((tmp_path / "runs/latest.json").read_text())
            assert record["status"] == "cancelled" and record["return_code"] == 130
            with scheduler.exclusive_lock(tmp_path / "runs/.pipeline.lock"):
                pass
        finally:
            scheduler.stop_process_tree(process, timeout=10)


def test_non_ascii_subprocess_output_is_logged_as_utf8(tmp_path, monkeypatch):
    text = "Подготовка → обучение → готово"
    monkeypatch.setattr(scheduler, "repro_command",
                        lambda _: [sys.executable, "-c", f"print({text!r})"])
    assert scheduler.run_once(tmp_path, train_only=True) == 0
    record = json.loads((tmp_path / "runs/latest.json").read_text())
    assert text in (tmp_path / record["log_file"]).read_text(encoding="utf-8")


def test_text_hash_accepts_only_line_ending_differences(tmp_path):
    import hashlib
    path = tmp_path / "sample.csv"
    lf = b"a,b\n1,2\n"
    crlf = lf.replace(b"\n", b"\r\n")
    for data in (lf, crlf):
        path.write_bytes(data)
        for expected in (lf, crlf):
            assert text_hash_matches(path, hashlib.sha256(expected).hexdigest())
        assert not text_hash_matches(path, hashlib.sha256(lf + b"\n").hexdigest())
