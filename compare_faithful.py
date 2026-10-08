"""Compare original Qwen-RLCD, faithful LoRA, and Jev on frozen banking data."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))

from mlx_lm import load

from banking_data import save as save_data
from faithful_rlcd import Faithful, evaluate, metrics
from mlx_jev_repro import SCHEMA

RUBRIC = [
    "Routine inquiry with no immediate disruption",
    "Access or payment disruption needing prompt attention",
    "Suspected ongoing unauthorized activity or imminent loss",
]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def read_key():
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is missing from the environment and .env")
    return key


def questions():
    return {
        "department": {
            "type": "choice",
            "instructions": SCHEMA[0]["question"],
            "criteria": {choice: None for choice in SCHEMA[0]["choices"]},
        },
        "urgency": {
            "type": "score",
            "instructions": SCHEMA[1]["question"],
            "criteria": RUBRIC,
        },
        "escalate": {
            "type": "noul",
            "instructions": SCHEMA[2]["question"],
        },
    }


def jev_probabilities(raw):
    answers = raw["answers"]
    distributions = [
        [answers["department"]["probabilities"][choice] for choice in SCHEMA[0]["choices"]],
        [answers["urgency"]["probabilities"][str(level)] for level in SCHEMA[1]["choices"]],
        [1.0 - answers["escalate"]["noul"], answers["escalate"]["noul"]],
    ]
    for distribution, spec in zip(distributions, SCHEMA):
        if len(distribution) != len(spec["choices"]):
            raise ValueError("Jev returned an unexpected candidate count")
        if min(distribution) < 0 or max(distribution) > 1:
            raise ValueError("Jev returned a probability outside [0, 1]")
        total = sum(distribution)
        if abs(total - 1) > 0.002:
            raise ValueError("Jev probabilities do not sum to one")
        for index in range(len(distribution)):
            distribution[index] /= total
    return distributions


def run_local(rows, model_path, adapter_path=None):
    model, tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
    model.eval()
    engine = Faithful(model, tokenizer)
    # Warm shaders and caches outside measured requests.
    engine.probabilities("Routine request for a recognized payment receipt.")
    return evaluate(engine, rows)


def run_jev(rows, model_name):
    records = []
    with httpx.Client(
        base_url="https://api.typesafe.ai",
        headers={"Authorization": "Bearer " + read_key()},
        timeout=30.0,
        follow_redirects=False,
    ) as client:
        for index, row in enumerate(rows):
            started = time.perf_counter()
            record = {
                "id": row["id"],
                "group": row["group"],
                "labels": row["labels"],
            }
            try:
                response = client.post(
                    "/v1/systemone",
                    json={
                        "state": row["state"],
                        "model": model_name,
                        "questions": questions(),
                    },
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Jev HTTP {response.status_code}")
                raw = response.json()
                record["probabilities"] = jev_probabilities(raw)
                record["raw"] = raw
            except Exception as error:
                # Do not persist credentials, headers, or provider error response bodies.
                record["error"] = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            records.append(record)
            print(f"Jev {index + 1}/{len(rows)}", record.get("error", "ok"), flush=True)
            if "error" in record:
                break  # Avoid repeated charges/retries after an uncertain failure.
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/faithful-comparison-v1")
    parser.add_argument("--model", default="models/qwen-1.5b-4bit")
    parser.add_argument("--adapter", default="runs/faithful-v1/best")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument(
        "--backend",
        choices=["all", "original", "ours", "jev"],
        default="all",
    )
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    data, manifest = save_data(ROOT / "runs/faithful-v1/data")
    rows = data["test"]
    dataset_hash = manifest["test"]["sha256"]

    selected = ["original", "ours", "jev"] if args.backend == "all" else [args.backend]
    if "original" in selected:
        records = run_local(rows, ROOT / args.model)
        dump(out / "original.json", {
            "backend": "unchanged-community-qwen-rlcd",
            "dataset_sha256": dataset_hash,
            "records": records,
            "summary": metrics(records),
        })
        print("original", json.dumps(metrics(records)), flush=True)
    if "ours" in selected:
        records = run_local(rows, ROOT / args.model, ROOT / args.adapter)
        dump(out / "ours.json", {
            "backend": "faithful-qwen-rlcd-lora",
            "dataset_sha256": dataset_hash,
            "adapter": args.adapter,
            "records": records,
            "summary": metrics(records),
        })
        print("ours", json.dumps(metrics(records)), flush=True)
    if "jev" in selected:
        records = run_jev(rows, args.jev_model)
        successful = [record for record in records if "probabilities" in record]
        summary = metrics(successful) if successful else {"cases": 0}
        summary["errors"] = len(records) - len(successful)
        dump(out / "jev.json", {
            "backend": "official-jev-api",
            "requested_model": args.jev_model,
            "dataset_sha256": dataset_hash,
            "records": records,
            "summary": summary,
        })
        print("jev", json.dumps(summary), flush=True)

    summaries = {}
    for name in ("original", "ours", "jev"):
        path = out / f"{name}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload["dataset_sha256"] != dataset_hash:
                raise ValueError("Results use different datasets; choose a new --out")
            summaries[name] = payload["summary"]
    dump(out / "summary.json", {
        "dataset_sha256": dataset_hash,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summaries": summaries,
    })


if __name__ == "__main__":
    main()
