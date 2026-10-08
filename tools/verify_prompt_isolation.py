#!/usr/bin/env python3
"""Verify that holdout labels and provenance never enter Nimble inference prompts."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--run", type=Path, default=Path("runs/banking-independent-holdout-v1"))
    args = ap.parse_args()
    root = args.root.resolve()
    run = args.run if args.run.is_absolute() else root / args.run
    sys.path.insert(0, str(root / "reference-nimble"))

    from transformers import AutoTokenizer
    from nimble.training.schema_data import as_scoring, encode_scoring

    tokenizer = AutoTokenizer.from_pretrained(str(root / "models/qwen-1.5b-4bit"), local_files_only=True)
    rows = [json.loads(line) for line in (run / "holdout.jsonl").read_text().splitlines()]
    for row in rows:
        original = encode_scoring(as_scoring(row, False), tokenizer, 768)
        for label in ("yes", "no", "insufficient_evidence"):
            changed = copy.deepcopy(row)
            changed["reference"] = {"target": label, "rationale": "DELIBERATE GOLD LEAK TEST"}
            changed["provenance"] = {"test": "DO NOT PUT IN PROMPT"}
            encoded = encode_scoring(as_scoring(changed, False), tokenizer, 768)
            assert encoded["prompt_token_ids"] == original["prompt_token_ids"]
            assert encoded["candidate_token_ids"] == original["candidate_token_ids"]
    print(f"PASS: {len(rows) * 3} prompt-isolation checks")


if __name__ == "__main__":
    main()
