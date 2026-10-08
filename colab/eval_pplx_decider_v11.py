#!/usr/bin/env python3
"""Evaluate Perplexity's released decider on the frozen 128-case comparison.

The checkpoint contains the exact inference source used for its published run.
This script imports that source from the snapshot rather than the older AutoJev
checkout used by our LoRA trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


REPO = "perplexity-ai/pplx-decider-v1.1-27b"
REVISION = "3b45dead91dfa6d95aad6b95764a606fab2bf7a6"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(raw: dict) -> dict:
    if "input" in raw:
        state = raw["input"]["state"]
        question = raw["input"]["questions"]["decision"]
        target = raw["reference"]["target"]
    else:
        state, question, target = raw["state"], raw["question"], raw["target"]
    return {
        "id": raw["id"],
        "family": raw.get("family", raw["id"]),
        "state": state,
        "question": question,
        "target": target,
    }


def labels(question: dict) -> list[object]:
    if question["type"] == "noul":
        return [False, True]
    if question["type"] == "score":
        return list(range(len(question["criteria"])))
    return list(question["criteria"])


def record(row: dict, probabilities: list[float], latency: float) -> dict:
    choices = labels(row["question"])
    probabilities = [float(value) for value in probabilities]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    target = bool(row["target"]) if row["question"]["type"] == "noul" else row["target"]
    prediction = choices[best]
    return {
        "id": row["id"],
        "family": row["family"],
        "type": row["question"]["type"],
        "labels": choices,
        "target": target,
        "prediction": prediction,
        "probabilities": probabilities,
        "confidence": probabilities[best],
        "correct": type(prediction) is type(target) and prediction == target,
        "latency_s": latency,
        "provider": REPO,
    }


def metrics(records: list[dict]) -> dict:
    brier: list[float] = []
    nll: list[float] = []
    bins: list[list[dict]] = [[] for _ in range(15)]
    for item in records:
        target = next(
            i for i, value in enumerate(item["labels"])
            if type(value) is type(item["target"]) and value == item["target"]
        )
        probabilities = item["probabilities"]
        brier.append(sum((value - int(i == target)) ** 2 for i, value in enumerate(probabilities)))
        nll.append(-math.log(max(probabilities[target], 1e-12)))
        bins[min(14, int(item["confidence"] * 15))].append(item)
    ece = sum(
        len(group) / len(records)
        * abs(
            sum(item["confidence"] for item in group) / len(group)
            - sum(item["correct"] for item in group) / len(group)
        )
        for group in bins if group
    )
    return {
        "count": len(records),
        "accuracy": sum(item["correct"] for item in records) / len(records),
        "brier": sum(brier) / len(brier),
        "nll": sum(nll) / len(nll),
        "ece": ece,
        "mean_latency_s": sum(item["latency_s"] for item in records) / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/content/pplx-decider-v1.1-27b"))
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/content/pplx-decider-v1.1-eval.json"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not (args.checkpoint / "decision_config.json").is_file():
        snapshot_download(REPO, revision=REVISION, local_dir=args.checkpoint)
    sys.path.insert(0, str(args.checkpoint / "source" / "src"))
    from autojev.model import DecisionModel  # pylint: disable=import-outside-toplevel

    rows = [normalize(row) for row in jsonl(args.eval_file)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    prior = json.loads(args.prior_report.read_text()) if args.prior_report else None
    if prior:
        prior_rows = prior.get("comparisons") or prior.get("pairs")
        if [row["id"] for row in rows] != [row["id"] for row in prior_rows]:
            raise ValueError("Frozen sample IDs differ from the prior comparison")

    print(json.dumps({"phase": "load", "checkpoint": str(args.checkpoint), "rows": len(rows)}), flush=True)
    model = DecisionModel(args.checkpoint, device="cuda")
    predictions: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        part = rows[start : start + args.batch_size]
        model_rows = [{"state": row["state"], "question": row["question"]} for row in part]
        began = time.perf_counter()
        distributions = model.predict(model_rows, batch_size=args.batch_size)
        elapsed = (time.perf_counter() - began) / len(part)
        predictions.extend(record(row, dist, elapsed) for row, dist in zip(part, distributions, strict=True))
        print(json.dumps({"phase": "inference", "progress": f"{len(predictions)}/{len(rows)}"}), flush=True)

    result = {
        "protocol": "Same frozen 128 banking cases used for our model, public AutoJev, and Jev",
        "repo": REPO,
        "revision": REVISION,
        "checkpoint_config": json.loads((args.checkpoint / "decision_config.json").read_text()),
        "eval_sha256": hashlib.sha256(args.eval_file.read_bytes()).hexdigest(),
        "sample_ids_verified_identical": bool(prior),
        "metrics": metrics(predictions),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "predictions": predictions,
    }
    if prior:
        result["prior"] = {
            "public_autojev": prior.get("public_autojev"),
            "our_banking_finetune": prior.get("our_banking_finetune", prior.get("ours")),
            "jev": prior.get("jev"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), **result["metrics"],
                      "peak_gpu_allocated_gib": result["peak_gpu_allocated_gib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
