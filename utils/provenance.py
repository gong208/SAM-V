#!/usr/bin/env python3
"""
Run provenance.

Every run directory records the git sha, the working-tree diff, the resolved
config, the exact command, and the environment it ran in, written by the runner
itself.  A run that cannot state its commit does not go in the paper.

Written as early as possible in a run, so a crash still leaves a directory that
says what was being attempted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - provenance must never kill a run
        return f"<git {' '.join(args)} failed: {exc}>"


def write_run_provenance(
    output_dir: Path,
    config: Mapping[str, Any],
    argv: Optional[Sequence[str]] = None,
) -> Path:
    """Write ``provenance.json`` + ``git_diff.patch`` into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = list(argv if argv is not None else sys.argv)
    diff = _git("diff", "HEAD")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD").strip(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        # --ignore-submodules=untracked: importing sam-hq writes __pycache__/ inside
        # that submodule, and plain `git status` counts untracked content inside a
        # submodule as a modification -- so without this every run reported dirty
        # and the flag carried no information. Still caught: modified tracked files
        # in the repo or in a submodule, and a submodule checked out at a different
        # commit than the one pinned.
        "git_dirty": bool(_git("status", "--porcelain",
                               "--ignore-submodules=untracked").strip()),
        "git_diff_bytes": len(diff),
        "command": " ".join(argv),
        "argv": argv,
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "config": dict(config),
    }
    with (output_dir / "provenance.json").open("w") as f:
        # default=str so a Path or any other repr-able value in the resolved config is
        # recorded rather than killing the run it is meant to document.
        json.dump(payload, f, indent=2, default=str)
    (output_dir / "git_diff.patch").write_text(diff)
    return output_dir / "provenance.json"
