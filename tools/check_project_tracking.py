#!/usr/bin/env python3
"""Block commits that skip project tracking updates."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable

TRACKING_FILES = {
    "changelog.md",
    "docs/project_status.md",
    "docs/decision_log.md",
}


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _normalized_paths(paths: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for path in paths:
        candidate = path.strip().replace("\\", "/")
        if candidate:
            normalized.add(candidate.lower())
    return normalized


def staged_files() -> set[str]:
    result = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMRTD"])
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Failed to inspect staged files."
        print(stderr, file=sys.stderr)
        raise RuntimeError("Unable to read staged files from git.")
    return _normalized_paths(result.stdout.splitlines())


def main() -> int:
    try:
        staged = staged_files()
    except RuntimeError:
        return 2

    if not staged:
        return 0

    if TRACKING_FILES.intersection(staged):
        return 0

    print("Commit blocked: project tracking update required.", file=sys.stderr)
    print("Update at least one of the following files before commit:", file=sys.stderr)
    for path in sorted(TRACKING_FILES):
        print(f"- {path}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Staged files detected:", file=sys.stderr)
    for path in sorted(staged):
        print(f"- {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
