"""Render/install a user service with the current checkout and Python paths."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pmldl.common import ROOT


def unit_quote(value: str) -> str:
    if any(character in value for character in ('\n', '\r', '\x00')):
        raise ValueError("Service paths must not contain newlines or NUL")
    # systemd expands % specifiers even in quotes. ExecStart expands $variables.
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def unit_workdir(value: str) -> str:
    # Unlike ExecStart, WorkingDirectory consumes a single raw path and does
    # not remove surrounding quotes. Preserve internal spaces; escape % only.
    if any(ord(character) < 32 for character in value) or value != value.strip() or value.endswith('\\'):
        raise ValueError("Unsupported control/trailing characters in service working directory")
    if not value.startswith("/"):
        raise ValueError("Service working directory must be absolute")
    return value.replace("%", "%%")


def render(root: Path, python: Path) -> str:
    template = (ROOT / "services/systemd/pmldl-pipeline.service").read_text()
    return (template.replace("@WORKDIR@", unit_workdir(str(root.absolute())))
            .replace("@PYTHON@", unit_quote(str(python.absolute())).replace('$', '$$'))
            .replace("@SCRIPT@", unit_quote(str(root.absolute() / "pipeline.py")).replace('$', '$$')))


def install(root: Path, python: Path, unit_dir: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.service", name):
        raise ValueError("Use a simple .service name without path separators")
    if not (root / "pipeline.py").is_file() or not python.is_file():
        raise ValueError("The checkout or Python interpreter does not exist")
    unit_dir.mkdir(parents=True, exist_ok=True)
    path = unit_dir / name
    path.write_text(render(root, python), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-dir", type=Path, default=Path.home() / ".config/systemd/user")
    parser.add_argument("--name", default="pmldl-pipeline.service")
    args = parser.parse_args()
    path = install(ROOT, Path(sys.executable), args.unit_dir, args.name)
    print(f"Installed {path}")
    print(f"systemctl --user daemon-reload\nsystemctl --user enable --now {args.name}")
    print("For operation after logout/reboot, enable user lingering and keep Docker running.")


if __name__ == "__main__":
    main()
