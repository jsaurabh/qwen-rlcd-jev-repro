"""Compare the local MLX decision adapter with Jev on public JevBench tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

ROOT = Path(__file__).resolve().parent
JEVBENCH = ROOT / "reference-jevbench"
NIMBLE = ROOT / "reference-nimble"
sys.path[:0] = [str(JEVBENCH), str(NIMBLE)]

from jevbench.adapters.typesafe import TypeSafeAdapter
from jevbench.scoring import score_task
from jevbench.summarize import summarize
from jevbench.tasks import dataset_hash, load_jsonl
from nimble.training.schema_data import as_scoring, encode_scoring
from nimble_1_5b_mlx import candidate_logits


TIERS = {
    "easy": JEVBENCH / "datasets/public/easy.jsonl",
    "standard": JEVBENCH / "datasets/public/original.jsonl",
    "hard": JEVBENCH / "datasets/public/hard.jsonl",
}


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def select_tasks(tiers: list[str], limit_per_tier: int | None):
    selected = []
    task_tier = {}
    for tier in tiers:
        rows = load_jsonl(str(TIERS[tier]))
        if limit_per_tier is not None:
            rows = rows[:limit_per_tier]
        selected.extend(rows)
        task_tier.update({row.id: tier for row in rows})
    return selected, task_tier


def raw_record(task) -> dict:
    target = task.expected
    if task.question["type"] == "noul":
        target = target == "yes"
    return {
        "id": task.id,
        "family": task.group or task.id,
        "source_family": task.family,
        "input": {"state": task.state, "questions": {"decision": task.question}},
        "reference": {"target": target},
    }


def result_record(task, tier, provider, started, probs=None, model="", error=None,
                  usage=None, status_code=None, prompt_tokens=None) -> dict:
    scored = score_task(probs or {}, task) if probs is not None else {
        "valid": False, "strict_valid": False, "renormalized": False,
        "correct": False, "predicted": None,
    }
    return {
        "task_id": task.id, "family": task.family, "tier": tier,
        "split": task.split, "group": task.group,
        "status": "ok" if probs is not None else "failed", "ok": probs is not None,
        "valid": scored["valid"], "strict_valid": scored.get("strict_valid", False),
        "renormalized": scored.get("renormalized", False),
        "correct": scored.get("correct", False), "predicted": scored.get("predicted"),
        "ordinal_ev": scored.get("ordinal_ev"), "probs": scored.get("probs"),
        "probs_as_returned": probs, "probs_source": "candidate_logits" if provider == "local" else "native",
        "model": model, "error": error, "schema_error": scored.get("error"),
        "status_code": status_code, "latency_s": time.perf_counter() - started,
        "usage": usage or {}, "prompt_tokens": prompt_tokens,
        "cost_usd": None, "cost_basis": "not_scored", "provider": provider,
    }


class LocalScorer:
    def __init__(self, model_path: Path, adapter_path: Path, max_tokens: int,
                 shuffle_seed: int | None = None):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.shuffle_seed = shuffle_seed
        adapter = None if str(adapter_path).lower() == "none" else str(adapter_path)
        self.model, self.tokenizer = load(str(model_path), adapter_path=adapter)
        self.model.eval()

    def run(self, task, tier):
        started = time.perf_counter()
        prompt_tokens = None
        try:
            encoded = encode_scoring(as_scoring(raw_record(task), False), self.tokenizer,
                                     self.max_tokens, self.shuffle_seed)
            prompt_tokens = len(encoded["prompt_token_ids"])
            row = {"input_ids": encoded["prompt_token_ids"], "candidate_ids": encoded["candidate_token_ids"]}
            logits = candidate_logits(self.model, row)
            mx.eval(logits)
            values = [str(v).lower() if isinstance(v, bool) else str(v)
                      for v in encoded["code_to_choice"].values()]
            if task.question["type"] == "noul":
                values = ["yes" if value == "true" else "no" for value in values]
            probs = dict(zip(values, map(float, mx.softmax(logits).tolist())))
            return result_record(task, tier, "local", started, probs=probs,
                                 model=str(self.model_path), prompt_tokens=prompt_tokens)
        except Exception as exc:
            return result_record(task, tier, "local", started,
                                 model=str(self.model_path), prompt_tokens=prompt_tokens,
                                 error=f"{type(exc).__name__}: {exc}")


class JevScorer:
    def __init__(self, model_name: str):
        self.adapter = TypeSafeAdapter(model=model_name, key_env="TYPESAFE_API_KEY", timeout_s=120)

    def run(self, task, tier):
        started = time.perf_counter()
        response = self.adapter.run(task)
        return result_record(
            task, tier, "jev", started, probs=response.probs if response.ok else None,
            model=response.model, error=response.error, usage=response.usage,
            status_code=response.status,
        )


def provider_summary(tasks, records, tiers):
    overall = summarize(tasks, records)
    overall["by_tier"] = {
        tier: summarize([t for t in tasks if any(r["task_id"] == t.id and r["tier"] == tier for r in records)],
                        [r for r in records if r["tier"] == tier])
        for tier in tiers
    }
    failures = {}
    for row in records:
        if not row["ok"]:
            key = (row.get("error") or "unknown").split(":", 1)[0]
            failures[key] = failures.get(key, 0) + 1
    overall["failure_types"] = failures
    return overall


def compact(summary):
    return {
        "attempted": summary["n_attempted"], "correct": summary["n_correct"],
        "accuracy": summary["accuracy"], "operational_success": summary["operational_success"],
        "ece": summary["ece"]["ece"] if summary.get("ece") else None,
        "brier": summary["brier_mean"], "p50_s": summary["latency"]["p50_s"],
        "p95_s": summary["latency"]["p95_s"], "failure_types": summary["failure_types"],
        "by_tier": {k: {"attempted": v["n_attempted"], "correct": v["n_correct"],
                         "accuracy": v["accuracy"], "operational_success": v["operational_success"]}
                    for k, v in summary["by_tier"].items()},
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", choices=("local", "jev", "both"), default="both")
    parser.add_argument("--tiers", default="easy,standard,hard")
    parser.add_argument("--limit-per-tier", type=int)
    parser.add_argument("--model", default="models/qwen-1.5b-4bit")
    parser.add_argument("--adapter", default="runs/banking-contrastive-v3/best")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--shuffle-seed", type=int,
                        help="deterministically permute candidates before local scoring")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]
    if not tiers or any(x not in TIERS for x in tiers):
        parser.error("--tiers must contain easy, standard, and/or hard")
    if args.limit_per_tier is not None and args.limit_per_tier <= 0:
        parser.error("--limit-per-tier must be positive")
    out = ROOT / args.out
    if out.exists():
        parser.error("output exists; choose a new --out")
    load_dotenv()
    tasks, task_tier = select_tasks(tiers, args.limit_per_tier)
    providers = ["local", "jev"] if args.providers == "both" else [args.providers]
    partial_dir = Path(str(out) + ".partial")
    run_contract = {
        "dataset_hash": dataset_hash(tasks), "tiers": tiers,
        "limit_per_tier": args.limit_per_tier, "providers": providers,
        "local_model": args.model, "local_adapter": args.adapter,
        "jev_model_requested": args.jev_model, "max_tokens": args.max_tokens,
        "shuffle_seed": args.shuffle_seed,
    }
    contract_path = partial_dir / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != run_contract:
        parser.error(f"partial run contract differs: {contract_path}")
    partial_dir.mkdir(parents=True, exist_ok=True)
    if not contract_path.exists():
        contract_path.write_text(json.dumps(run_contract, indent=2, sort_keys=True) + "\n")
    row_paths = {name: partial_dir / f"{name}.jsonl" for name in providers}
    rows = {name: read_jsonl(row_paths[name]) for name in providers}
    for name, records in rows.items():
        ids = [row["task_id"] for row in records]
        if len(ids) != len(set(ids)) or not set(ids) <= {task.id for task in tasks}:
            parser.error(f"invalid partial results: {row_paths[name]}")
    scorers = {}
    if "local" in providers and len(rows["local"]) < len(tasks):
        adapter_path = Path("none") if args.adapter.lower() == "none" else ROOT / args.adapter
        scorers["local"] = LocalScorer(ROOT / args.model, adapter_path, args.max_tokens,
                                       args.shuffle_seed)
    if "jev" in providers and len(rows["jev"]) < len(tasks):
        if not os.environ.get("TYPESAFE_API_KEY"):
            parser.error("TYPESAFE_API_KEY is missing from the environment and .env")
        scorers["jev"] = JevScorer(args.jev_model)
    started = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        for name in providers:
            if any(row["task_id"] == task.id for row in rows[name]):
                continue
            row = scorers[name].run(task, task_tier[task.id])
            rows[name].append(row)
            append_jsonl(row_paths[name], row)
            print(json.dumps({"progress": f"{i}/{len(tasks)}", "provider": name,
                              "task": task.id, "ok": row["ok"],
                              "correct": row["correct"], "prediction": row["predicted"]}), flush=True)
    summaries = {name: provider_summary(tasks, records, tiers) for name, records in rows.items()}
    comparison = None
    if set(providers) == {"local", "jev"}:
        left = {r["task_id"]: r for r in rows["local"]}
        right = {r["task_id"]: r for r in rows["jev"]}
        paired = [(left[t.id], right[t.id]) for t in tasks]
        comparison = {
            "count": len(paired),
            "prediction_agreement": sum(a["predicted"] == b["predicted"] for a, b in paired) / len(paired),
            "both_correct": sum(a["correct"] and b["correct"] for a, b in paired),
            "local_only_correct": sum(a["correct"] and not b["correct"] for a, b in paired),
            "jev_only_correct": sum(not a["correct"] and b["correct"] for a, b in paired),
            "neither_correct": sum(not a["correct"] and not b["correct"] for a, b in paired),
        }
    commit = "unknown"
    head = JEVBENCH / ".git/HEAD"
    if head.exists():
        commit = subprocess.run(["git", "-C", str(JEVBENCH), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "JevBench public tasks; same state/question/criteria/choice order",
        "jevbench_commit": commit, "dataset_hash": dataset_hash(tasks),
        "tiers": tiers, "limit_per_tier": args.limit_per_tier,
        "providers": providers, "local_model": args.model, "local_adapter": args.adapter,
        "jev_model_requested": args.jev_model, "max_tokens": args.max_tokens,
        "shuffle_seed": args.shuffle_seed,
        "credentials_saved": False, "elapsed_s": time.perf_counter() - started,
        "versions": {"python": platform.python_version(),
                     **{k: version(k) for k in ("mlx", "mlx-lm")}},
        "summary": {name: compact(value) for name, value in summaries.items()},
        "comparison": comparison, "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), "summary": payload["summary"],
                      "comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
