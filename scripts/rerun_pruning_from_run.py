#!/usr/bin/env python3
"""
Deprecated alias for `scripts/extend_run.py`.

This wrapper preserves the old CLI while routing to the new, more general tool.
It forces `--tasks pruning` unless the caller explicitly provided `--tasks`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    argv = list(sys.argv[1:])
    if "--tasks" not in argv:
        argv = ["--tasks", "pruning"] + argv
    sys.argv = [sys.argv[0]] + argv
    target = Path(__file__).resolve().parent / "extend_run.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

