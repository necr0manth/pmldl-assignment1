"""Run the complete DVC graph immediately and every five minutes thereafter.

Linux/macOS/WSL2 only: fcntl locks prevent overlapping local processes.
Use the supplied systemd user service for persistent scheduling.
"""
from __future__ import annotations

import argparse
import fcntl
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from pmldl.common import ROOT, utc_now, write_json
from pmldl.deployment import compose_command


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AlreadyRunning(f"Another process holds {path.name}") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()))
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


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
                env["DVC_NO_ANALYTICS"] = "1"
                # Stage commands use `python`: put this interpreter's venv first.
                env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
                env["PYTHONPATH"] = str(root / "code") + os.pathsep + env.get("PYTHONPATH", "")
                process = subprocess.Popen(repro_command(train_only), cwd=root, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True)
                for line in process.stdout:
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
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
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
