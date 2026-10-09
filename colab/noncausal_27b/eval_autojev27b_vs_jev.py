#!/usr/bin/env python3
"""Evaluate the completed 27B AutoJev LoRA and TypeSafe Jev on identical banking cases."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file

from autojev.model import DecisionModel, options


API_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = {0, 429, 500, 502, 503, 504, 529}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, help="Completed selected artifact; auto-discovered when omitted")
    p.add_argument("--eval-file", type=Path, help="Banking JSONL; auto-discovered when omitted")
    p.add_argument("--output", type=Path,
                   default=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b/eval-v1"))
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--jev-model", default="jev-1.13.0")
    p.add_argument("--local-only", action="store_true")
    return p.parse_args()


def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def discover_artifact(root=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b")):
    candidates = []
    for config_path in root.glob("**/selected/decision_config.json"):
        try:
            cfg = json.loads(config_path.read_text())
            if (config_path.parent / "adapter").is_dir() and (config_path.parent / "readout.safetensors").is_file():
                candidates.append((int(cfg.get("steps", cfg.get("step", -1))), config_path.stat().st_mtime,
                                   config_path.parent))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    if not candidates:
        raise FileNotFoundError(f"No completed selected artifact found below {root}")
    return max(candidates)[2]


def discover_eval_file():
    candidates = [
        Path("/content/jev_gap_curriculum_v1/eval.jsonl"),
        Path("/content/autojev-data/eval.jsonl"),
        Path("/content/autojev-data/dev.jsonl"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Banking eval JSONL not found; expected /content/jev_gap_curriculum_v1/eval.jsonl")


def normalize_row(raw):
    if "input" in raw:
        question = raw["input"]["questions"]["decision"]
        state = raw["input"]["state"]
        target = raw["reference"]["target"]
    else:
        question, state, target = raw["question"], raw["state"], raw["target"]
    return {
        "id": raw["id"], "domain": raw.get("domain", "banking"),
        "family": raw.get("family", raw["id"]), "state": state,
        "question": question, "target": target,
    }


def load_model(artifact):
    cfg = json.loads((artifact / "decision_config.json").read_text())
    model = DecisionModel(train=False, base_model=cfg["base_model"], revision=cfg["revision"])
    mode = cfg.get("attention_mode", "causal")
    if mode == "noncausal_full_attention":
        from autojev_lora_train import enable_noncausal_full_attention
        enable_noncausal_full_attention(model.backbone.language_model)
    elif mode != "causal":
        raise ValueError(f"Unknown saved attention mode: {mode}")
    model.backbone = PeftModel.from_pretrained(model.backbone, artifact / "adapter").eval()
    model.readout.load_state_dict(load_file(str(artifact / "readout.safetensors")))
    model.processor = model.processor.from_pretrained(str(artifact / "processor"))
    model.processor.tokenizer.padding_side = "left"
    model.temperature = float(cfg["temperature"])
    model.eval()
    return model, cfg


def label_values(question):
    keys, _ = options(question)
    if question["type"] == "noul":
        return [False, True]
    if question["type"] == "score":
        return list(range(len(keys)))
    return keys


def prediction(row, probs, latency_s, provider, extra=None):
    labels = label_values(row["question"])
    if len(labels) != len(probs):
        raise ValueError("Probability width mismatch")
    total = sum(float(x) for x in probs)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid probability mass")
    probs = [float(x) / total for x in probs]
    best = max(range(len(probs)), key=probs.__getitem__)
    target = row["target"]
    if row["question"]["type"] == "noul":
        target = bool(target)
    result = {
        "id": row["id"], "family": row["family"], "type": row["question"]["type"],
        "input_sha256": sha({"state": row["state"], "question": row["question"]}),
        "labels": labels, "target": target, "prediction": labels[best],
        "probabilities": probs, "confidence": probs[best],
        "correct": type(labels[best]) is type(target) and labels[best] == target,
        "latency_s": latency_s, "provider": provider,
    }
    if extra:
        result.update(extra)
    return result


def local_predictions(model, rows, batch_size):
    result = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            model_rows = [{"state": r["state"], "question": r["question"]} for r in part]
            began = time.perf_counter()
            distributions = model.predict(model_rows, batch_size=batch_size)
            elapsed = time.perf_counter() - began
            result.extend(prediction(r, p, elapsed / len(part), "autojev-27b")
                          for r, p in zip(part, distributions, strict=True))
            print(json.dumps({"phase": "local", "progress": f"{len(result)}/{len(rows)}"}), flush=True)
    return result


def jev_payload(row, model):
    return {"model": model, "state": row["state"], "questions": {"decision": row["question"]}}


def jev_probs(question, answer):
    kind = question["type"]
    if kind == "noul":
        value = float(answer["noul"])
        return [1.0 - value, value]
    keys = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
    return [float(answer["probabilities"][key]) for key in keys]


def call_jev(row, model, key, attempts=5):
    payload = jev_payload(row, model)
    data = json.dumps(payload, allow_nan=False).encode()
    started = time.perf_counter()
    for attempt in range(attempts):
        request = urllib.request.Request(API_URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = json.load(response)
            answer = raw["answers"]["decision"]
            return prediction(row, jev_probs(row["question"], answer), time.perf_counter() - started,
                              raw.get("model", model), {"provider_confidence": answer.get("confidence"),
                                                        "usage": raw.get("usage")})
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (401, 403):
                raise RuntimeError("TypeSafe rejected TYPESAFE_API_KEY") from None
            if status not in RETRYABLE or attempt + 1 == attempts:
                raise RuntimeError(f"TypeSafe HTTP {status}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt + 1 == attempts:
                raise RuntimeError("TypeSafe connection failed") from None
        time.sleep(2 ** attempt)


def get_colab_secret():
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    try:
        from google.colab import userdata
        return userdata.get("TYPESAFE_API_KEY")
    except Exception as exc:
        raise RuntimeError("Add TYPESAFE_API_KEY to Colab Secrets and enable notebook access") from exc


def jev_predictions(rows, model, workers, output):
    cache_path = output / "jev-cache.jsonl"
    cached = {}
    if cache_path.is_file():
        for item in jsonl(cache_path):
            cached[item["id"]] = item
    key = get_colab_secret()
    pending = [row for row in rows if row["id"] not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_jev, row, model, key): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {"id": row["id"], "error": f"{type(exc).__name__}: {exc}", "provider": model}
            cached[row["id"]] = item
            with cache_path.open("a") as f:
                f.write(json.dumps(item, allow_nan=False) + "\n")
            print(json.dumps({"phase": "jev", "progress": f"{len(cached)}/{len(rows)}",
                              "errors": sum("error" in x for x in cached.values())}), flush=True)
    return [cached[row["id"]] for row in rows]


def metrics(records):
    valid = [r for r in records if "error" not in r]
    if not valid:
        return {"count": len(records), "valid": 0}
    brier, nll = [], []
    bins = [[] for _ in range(15)]
    for r in valid:
        target = next(i for i, label in enumerate(r["labels"])
                      if type(label) is type(r["target"]) and label == r["target"])
        p = r["probabilities"]
        brier.append(sum((value - int(i == target)) ** 2 for i, value in enumerate(p)))
        nll.append(-math.log(max(p[target], 1e-12)))
        bins[min(14, int(r["confidence"] * 15))].append(r)
    ece = 0.0
    for group in bins:
        if group:
            ece += len(group) / len(valid) * abs(
                sum(x["confidence"] for x in group) / len(group) -
                sum(x["correct"] for x in group) / len(group))
    return {
        "count": len(records), "valid": len(valid),
        "accuracy": sum(r["correct"] for r in valid) / len(valid),
        "brier": sum(brier) / len(brier), "nll": sum(nll) / len(nll), "ece": ece,
        "mean_latency_s": sum(r["latency_s"] for r in valid) / len(valid),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact or discover_artifact()
    eval_file = args.eval_file or discover_eval_file()
    rows = [normalize_row(r) for r in jsonl(eval_file)]
    rows = [r for r in rows if r["question"]["type"] in ("choice", "noul", "score")]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    print(json.dumps({"artifact": str(artifact), "eval_file": str(eval_file), "cases": len(rows)}), flush=True)
    model, config = load_model(artifact)
    local = local_predictions(model, rows, args.batch_size)
    (args.output / "local.json").write_text(json.dumps(local, indent=2, allow_nan=False) + "\n")
    del model
    torch.cuda.empty_cache()
    jev = [] if args.local_only else jev_predictions(rows, args.jev_model, args.workers, args.output)
    jev_by_id = {r["id"]: r for r in jev}
    pairs = []
    for ours in local:
        official = jev_by_id.get(ours["id"])
        if official and "error" not in official:
            pairs.append({"id": ours["id"], "target": ours["target"], "ours": ours,
                          "jev": official, "agree": ours["prediction"] == official["prediction"]})
    report = {
        "protocol": "Identical held-out banking inputs; references joined only after inference",
        "created_unix": time.time(), "artifact": str(artifact), "artifact_config": config,
        "eval_file": str(eval_file), "eval_sha256": hashlib.sha256(eval_file.read_bytes()).hexdigest(),
        "seed": args.seed, "limit": args.limit, "jev_model_requested": args.jev_model,
        "ours": metrics(local), "jev": metrics(jev) if jev else None,
        "paired": {"count": len(pairs), "agreement": (sum(p["agree"] for p in pairs) / len(pairs)) if pairs else None},
        "pairs": pairs,
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output / "comparison.json"), "ours": report["ours"],
                      "jev": report["jev"], "paired": report["paired"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
