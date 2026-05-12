"""Orchestrator that runs all make_fig_*.py scripts in this repo.

This script discovers every `make_fig_*.py` file under `src/` and
runs them in sorted order. Each individual script is responsible
for its own data loading + matplotlib output to `paper/figures/`.

Usage:
    python ./src/make_figures.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


def main() -> int:
    scripts = sorted(SRC.glob("make_fig_*.py"))
    if not scripts:
        print(f"No make_fig_*.py scripts found in {SRC}")
        return 0
    failed = []
    for s in scripts:
        print(f"--- running {s.name} ---")
        try:
            subprocess.check_call([sys.executable, str(s)], cwd=REPO)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: rc={e.returncode}")
            failed.append(s.name)
    print()
    print(f"Ran {len(scripts)} figure scripts, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
