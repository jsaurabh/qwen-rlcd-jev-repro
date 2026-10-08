#!/usr/bin/env python3
"""Merge AutoJev's public text corpus with the frozen gap curriculum and filter by tokens."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import string
from pathlib import Path

from transformers import AutoProcessor

from autojev.model import decision_messages


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def convert_gap(row: dict) -> dict:
    question = row["input"]["questions"]["decision"]
    label = row["reference"]["target"]
    return {
        "id": f"gap:{row['id']}",
        "suite": "jev-gap-curriculum-v1",
        "family": row["family"],
        "state": row["input"]["state"],
        "question": question,
        "target": label,
        "label": label,
        "source": {
            "dataset": "jev-gap-curriculum-v1",
            "split": row["split"],
            "license": "MIT",
            "generator": "deterministic_rule_engine",
        },
    }


def answer_codes(tokenizer) -> list[str]:
    candidates = list(string.ascii_uppercase) + [
        "".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    codes = [code for code in candidates if len(tokenizer.encode(code, add_special_tokens=False)) == 1][:255]
    if len(codes) != 255:
        raise ValueError("Tokenizer does not provide 255 one-token answer codes")
    return codes


def token_count(row: dict, processor, codes: list[str]) -> int:
    messages = decision_messages(row, codes)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return len(processor.tokenizer.encode(text, add_special_tokens=False))


def filtered(rows: list[dict], processor, codes: list[str], maximum: int) -> tuple[list[dict], int]:
    kept = []
    observed = 0
    for row in rows:
        count = token_count(row, processor, codes)
        observed = max(observed, count)
        if count <= maximum:
            kept.append(row)
    return kept, observed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--autojev-data", type=Path, default=Path("/content/autojev-data"))
    ap.add_argument("--gap-data", type=Path, default=Path("/content/qwen-rlcd-jev/data/jev_gap_curriculum_v1"))
    ap.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.base_model, revision=args.revision)
    codes = answer_codes(processor.tokenizer)
    original = read(args.autojev_data / "train.jsonl")
    gap = [convert_gap(row) for row in read(args.gap_data / "train.jsonl")]
    train = original + gap
    random.Random(args.seed).shuffle(train)

    sources = {
        "train-mixed.jsonl": train,
        "dev-filtered.jsonl": read(args.autojev_data / "dev.jsonl"),
        "temperature-filtered.jsonl": read(args.autojev_data / "temperature.jsonl"),
    }
    report = []
    for name, rows in sources.items():
        kept, maximum = filtered(rows, processor, codes, args.max_length)
        write(args.autojev_data / name, kept)
        report.append({
            "file": name,
            "original": len(rows),
            "kept": len(kept),
            "rejected": len(rows) - len(kept),
            "retained_percent": round(100 * len(kept) / len(rows), 2),
            "maximum_tokens": maximum,
        })
    (args.autojev_data / "filter-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
