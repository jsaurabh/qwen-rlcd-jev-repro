#!/usr/bin/env python3
"""Fast integrity and publication-safety checks for this repository."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 10 * 1024 * 1024
FORBIDDEN_PARTS = {".env", ".venv", "models", "runs", "colab-results", "__pycache__"}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style secret": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile(rb"AIza[A-Za-z0-9_-]{20,}"),
    "assigned API secret": re.compile(rb"(?i)(?:api[_-]?key|token|secret)\s*[=:]\s*['\"]?[A-Za-z0-9_./+-]{24,}"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True,
        ).stdout
        return [ROOT / item.decode() for item in result.split(b"\0") if item]
    return [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def check_data() -> None:
    gap = json.loads((ROOT / "data/jev_gap_curriculum_v1/manifest.json").read_text())
    all_path = ROOT / "data/jev_gap_curriculum_v1/all.jsonl"
    assert sha(all_path) == gap["sha256"]
    assert sum(1 for _ in all_path.open()) == gap["rows"]

def check_manifest() -> None:
    manifest_path = ROOT / "PUBLIC_MANIFEST.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected = {}
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    current = {
        str(path.relative_to(ROOT)): sha(path)
        for path in tracked()
        if path != manifest_path
    }
    assert expected == current, "PUBLIC_MANIFEST.sha256 does not match tracked files"


def main() -> None:
    errors: list[str] = []
    files = tracked()
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden tracked path: {relative}")
            continue
        size = path.stat().st_size
        if size > MAX_FILE:
            errors.append(f"file exceeds 10 MiB: {relative} ({size} bytes)")
        data = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                errors.append(f"possible {label}: {relative}")

        if path.suffix == ".json":
            try:
                json.loads(data)
            except Exception as exc:
                errors.append(f"invalid JSON {relative}: {exc}")
        elif path.suffix == ".jsonl":
            for line_number, line in enumerate(data.splitlines(), 1):
                try:
                    json.loads(line)
                except Exception as exc:
                    errors.append(f"invalid JSONL {relative}:{line_number}: {exc}")
                    break
        elif path.suffix == ".ipynb":
            notebook = json.loads(data)
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                    errors.append(f"notebook contains saved output: {relative} cell {index}")

    try:
        check_data()
    except Exception as exc:
        errors.append(f"dataset integrity: {exc}")
    try:
        check_manifest()
    except Exception as exc:
        errors.append(f"public manifest: {exc}")

    if errors:
        print("Publication checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Publication checks passed for {len(files)} tracked files.")


if __name__ == "__main__":
    main()
