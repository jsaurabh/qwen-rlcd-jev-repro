#!/usr/bin/env python3
"""Evaluate the released full-weight AutoJev-27B on the exact prior 128-case sample."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from autojev.model import DecisionModel


def load_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("comparison_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="denis-pplx/autojev-27b")
    ap.add_argument("--checkpoint", type=Path, default=Path("/content/autojev-public-27b"))
    ap.add_argument("--eval-file", type=Path, default=Path("/content/jev_gap_curriculum_v1/eval.jsonl"))
    ap.add_argument("--previous", type=Path,
                    default=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b/eval-v1/comparison.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b/eval-v1"))
    ap.add_argument("--helpers", type=Path,
                    default=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b/full-v4/eval_autojev27b_vs_jev.py"))
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    helpers = load_helpers(args.helpers)
    previous = json.loads(args.previous.read_text())
    expected_ids = [pair["id"] for pair in previous["pairs"]]
    if len(expected_ids) != args.limit:
        raise ValueError(f"Previous report contains {len(expected_ids)} paired rows, expected {args.limit}")

    token = os.environ.get("HF_TOKEN") or None
    print(json.dumps({"phase": "download", "repo": args.repo, "destination": str(args.checkpoint)}), flush=True)
    snapshot_download(repo_id=args.repo, local_dir=args.checkpoint, token=token)

    rows = [helpers.normalize_row(r) for r in helpers.jsonl(args.eval_file)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    actual_ids = [row["id"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("Dataset/sample IDs differ from the completed ours-vs-Jev run")

    print(json.dumps({"phase": "load", "checkpoint": str(args.checkpoint)}), flush=True)
    model = DecisionModel(checkpoint=args.checkpoint, train=False)
    started = time.perf_counter()
    public = helpers.local_predictions(model, rows, args.batch_size)
    elapsed = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()

    public_path = args.output / "public-autojev.json"
    public_path.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
    previous_by_id = {pair["id"]: pair for pair in previous["pairs"]}
    comparisons = []
    for baseline in public:
        prior = previous_by_id[baseline["id"]]
        comparisons.append({
            "id": baseline["id"], "target": baseline["target"],
            "public_autojev": baseline,
            "our_banking_finetune": prior["ours"],
            "jev": prior["jev"],
        })

    def wins(left, right):
        return sum(a[left]["correct"] and not a[right]["correct"] for a in comparisons)

    report = {
        "protocol": previous["protocol"],
        "sample_ids_verified_identical": True,
        "public_autojev_repo": args.repo,
        "public_autojev_checkpoint": str(args.checkpoint),
        "public_autojev": helpers.metrics(public),
        "our_banking_finetune": previous["ours"],
        "jev": previous["jev"],
        "pairwise_unique_correct": {
            "public_over_ours": wins("public_autojev", "our_banking_finetune"),
            "ours_over_public": wins("our_banking_finetune", "public_autojev"),
            "public_over_jev": wins("public_autojev", "jev"),
            "jev_over_public": wins("jev", "public_autojev"),
        },
        "inference_wall_seconds": elapsed,
        "comparisons": comparisons,
    }
    out = args.output / "three-way-comparison.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), **{k: report[k] for k in
        ("public_autojev", "our_banking_finetune", "jev", "pairwise_unique_correct")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
