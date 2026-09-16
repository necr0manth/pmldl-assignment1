"""Convenient project entry point; also works before an editable install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from pmldl.scheduler import main

if __name__ == "__main__":
    raise SystemExit(main())
