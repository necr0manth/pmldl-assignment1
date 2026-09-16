"""Run the complete DVC graph immediately and every five minutes thereafter.

Windows, Linux and macOS: native file locks prevent overlapping local processes.
The optional systemd user service is Linux-only.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from filelock import FileLock, Timeout

from pmldl.common import ROOT, utc_now, write_json
from pmldl.deployment import compose_command


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def exclusive_lock(path: Path):
    """Fail fast using a native OS lock, not the existence of a stale file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path, timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise AlreadyRunning(f"Another process holds {path.name}") from error
    try:
        yield
    finally:
        lock.release()


def process_group_options() -> dict:
    """Isolate the child tree from terminal Ctrl+C; the runner owns cleanup."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_process_tree(process: subprocess.Popen, timeout: float = 30) -> None:
    """Stop the DVC process and its children, never unrelated Python processes.

    Windows uses taskkill /T /F (a forced stop); POSIX first sends SIGTERM to
    the dedicated process group and escalates to SIGKILL on timeout.
    Docker containers themselves remain managed by the Docker daemon.
    """
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if result.returncode and process.poll() is None:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, output=result.stdout, stderr=result.stderr
            )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # The child may have exited between poll() and killpg().
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=timeout)


def output_lines(process: subprocess.Popen):
    """Keep the main thread interruptible while a silent Windows child runs.

    A blocking pipe readline in the main thread can delay Ctrl+C on Windows.
    Read in a daemon thread and poll a queue with a short finite timeout instead.
    """
    queue = Queue()

    def read_output():
        try:
            for line in process.stdout:
                queue.put(line)
        except Exception as error:
            queue.put(error)
        finally:
            queue.put(None)

    Thread(target=read_output, name="pipeline-output", daemon=True).start()
    while True:
        try:
            value = queue.get(timeout=0.1)
        except Empty:
            continue
        if value is None:
            return
        if isinstance(value, Exception):
            raise value
        yield value


def repro_command(train_only: bool = False) -> list[str]:
    return [sys.executable, "-m", "dvc", "repro", "--force", "--no-run-cache",
            "train" if train_only else "deploy"]


def next_slot(start: float, finished: float, interval: float) -> float:
    """Skip missed slots; do not overlap runs or accumulate catch-up work."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    return start + max(1, math.floor((finished - start) / interval) + 1) * interval


def run_once(root: Path = ROOT, *, train_only: bool = False) -> int:
    with exclusive_lock(root / "runs/.pipeline.lock"):
        run_id = uuid.uuid4().hex
        record = {"run_id": run_id, "started_at": utc_now(),
                  "target": "train" if train_only else "deploy", "status": "running"}
        started = time.monotonic()
        log_path = root / "runs" / f"{run_id}.log"
        record["log_file"] = str(log_path.relative_to(root))
        write_json(root / "runs/latest.json", record)
        process = None
        return_code = 1
        with log_path.open("w", encoding="utf-8") as log:
            try:
                if not train_only:
                    subprocess.run(["docker", "info"], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
                    subprocess.run(["docker", "compose", "version"], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"
                env["DVC_NO_ANALYTICS"] = "1"
                # Stage commands use `python`: put this interpreter's venv first.
                env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
                env["PYTHONPATH"] = str(root / "code") + os.pathsep + env.get("PYTHONPATH", "")
                process = subprocess.Popen(repro_command(train_only), cwd=root, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace",
                                           bufsize=1, **process_group_options())
                for line in output_lines(process):
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
                record["status"] = "succeeded" if return_code == 0 else "failed"
            except KeyboardInterrupt:
                record["status"] = "cancelled"
                return_code = 130
                raise
            except (OSError, subprocess.SubprocessError) as error:
                record["status"] = "failed"
                record["error"] = str(error)
                print(f"Pipeline failed: {error}", file=sys.stderr)
                log.write(f"Pipeline failed: {error}\n")
            finally:
                if process is not None:
                    stop_process_tree(process)
                    if process.stdout:
                        process.stdout.close()
                record.update(finished_at=utc_now(), return_code=return_code,
                              duration_seconds=round(time.monotonic() - started, 3))
                write_json(root / "runs" / f"{run_id}.json", record)
                write_json(root / "runs/latest.json", record)
        return return_code


def schedule(root: Path = ROOT, *, interval: int = 300, max_runs: int | None = None) -> None:
    if interval < 300:
        raise ValueError("The interval must be at least 300 seconds")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    with exclusive_lock(root / "runs/.scheduler.lock"):
        count = 0
        while True:
            started = time.monotonic()
            print(f"[{utc_now()}] Starting complete pipeline", flush=True)
            try:
                code = run_once(root)
                print(f"[{utc_now()}] Pipeline finished with exit code {code}", flush=True)
            except AlreadyRunning as error:
                print(f"Skipping occupied slot: {error}", flush=True)
            count += 1
            if max_runs is not None and count >= max_runs:
                return
            delay = max(0.0, next_slot(started, time.monotonic(), interval) - time.monotonic())
            print(f"Next scheduled attempt in {delay:.1f} seconds", flush=True)
            time.sleep(delay)


def stop_requested(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run all three stages once")
    run.add_argument("--train-only", action="store_true", help="Developer mode: omit Docker deployment")
    recurring = commands.add_parser("schedule", help="Run now and every five minutes")
    recurring.add_argument("--interval", type=int, default=300)
    recurring.add_argument("--max-runs", type=int, default=None)
    commands.add_parser("status", help="Show the last pipeline result")
    commands.add_parser("down", help="Stop and remove this project's API/app containers")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, stop_requested)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, stop_requested)
        # Redirected PowerShell output need not use UTF-8 by default.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        if args.command == "run":
            return run_once(train_only=args.train_only)
        if args.command == "schedule":
            schedule(interval=args.interval, max_runs=args.max_runs)
        elif args.command == "status":
            path = ROOT / "runs/latest.json"
            print(path.read_text() if path.exists() else "No pipeline runs yet.")
        elif args.command == "down":
            # Refuse to race a running scheduler, which would recreate containers.
            with exclusive_lock(ROOT / "runs/.scheduler.lock"):
                with exclusive_lock(ROOT / "runs/.pipeline.lock"):
                    subprocess.run([*compose_command(), "down"], check=True)
        return 0
    except AlreadyRunning as error:
        print(str(error), file=sys.stderr)
        return 75
    except KeyboardInterrupt:
        print("Stopped. Existing deployed containers are left running.", file=sys.stderr)
        return 130
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
