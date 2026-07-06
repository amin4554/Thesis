#!/usr/bin/env python3
"""Pre-push safety check for the reviewer-facing thesis repository."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "PROJECT_STATUS.md",
    "THESIS_PLAN.md",
    "FROZEN_PROTOCOL.md",
    "requirements.txt",
    "src/protocol.py",
    "src/data_prep.py",
    "src/tracking.py",
    "tests/test_data_prep.py",
    "code_and_models/REPRODUCE_TRAINING.md",
}
FORBIDDEN_PARTS = {".claude", ".cursor", ".idea", ".vscode", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".jsonl",
    ".parquet",
    ".sqlite",
    ".docx",
    ".pptx",
    ".pdf",
    ".safetensors",
    ".ckpt",
    ".onnx",
}
MAX_TRACKED_BYTES = 20 * 1024 * 1024


def git(*args: str) -> str:
    cmd = [
        "git",
        "-c",
        f"safe.directory={ROOT.as_posix()}",
        "-C",
        str(ROOT),
        *args,
    ]
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout


def main() -> int:
    errors: list[str] = []
    tracked = {line.strip() for line in git("ls-files").splitlines() if line.strip()}

    for path in sorted(REQUIRED - tracked):
        errors.append(f"required file is not tracked: {path}")

    for rel in sorted(tracked):
        path = Path(rel)
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts & FORBIDDEN_PARTS:
            errors.append(f"private/cache path is tracked: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"large/binary artifact is tracked: {rel}")
        full = ROOT / path
        if full.is_file() and full.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 20 MiB: {rel}")

    if errors:
        print("Repository check FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Repository check passed: {len(tracked)} tracked files are safe to push.")
    dirty = git("status", "--short").strip()
    if dirty:
        print("Working tree changes still need review/commit:\n" + dirty)
    else:
        print("Working tree is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
