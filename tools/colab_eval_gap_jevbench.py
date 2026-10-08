import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time


NIMBLE_COMMIT = "f136b3f75721fda4ea961f73993cc50b08488835"
JEVBENCH_COMMIT = "5e95f23cbb7be098a9061fea924c4421620ab1a5"
ARCHIVE_SHA256 = "1613407ae9f91b1b8b7197e75a1b35a3fa44f91a1c7c326e348a3e43d3d30d2c"
root = pathlib.Path("/content")
nimble = root / "nimble"
jevbench = root / "jevbench"
adapter = root / "nimble-gap-9b-v1"
archive = root / "nimble-gap-9b-v1-adapter.tar.gz"
output = root / "nimble-gap-9b-v1-jevbench-public.json"


def run(*args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


for path, url, commit in ((nimble, "https://github.com/bespokelabsai/nimble.git", NIMBLE_COMMIT), (jevbench, "https://github.com/fstandhartinger/jevbench.git", JEVBENCH_COMMIT)):
    if not path.exists(): run("git", "clone", url, path)
    run("git", "fetch", "origin", commit, cwd=path)
    run("git", "checkout", "--detach", commit, cwd=path)

sys.path[:0] = [str(nimble), str(jevbench)]
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from jevbench.scoring import score_task
from jevbench.summarize import summarize
from jevbench.tasks import dataset_hash, load_jsonl
from nimble.training.model_loading import load_base
from nimble.training.schema_data import as_scoring, encode_scoring
from nimble.training.schema_train import CandidateCollator, candidate_logits

contract = json.loads((adapter / "schema_config.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(adapter)
tokenizer.padding_side = "left"
base = load_base(contract["model"], contract["revision"])
model = PeftModel.from_pretrained(base, adapter).eval()
collator = CandidateCollator(tokenizer.pad_token_id)

tiers = {
    "easy": jevbench / "datasets/public/easy.jsonl",
    "standard": jevbench / "datasets/public/original.jsonl",
    "hard": jevbench / "datasets/public/hard.jsonl",
}
tasks = []
task_tier = {}
for tier, path in tiers.items():
    rows = load_jsonl(str(path))
    tasks.extend(rows)
    task_tier.update({row.id: tier for row in rows})


def raw_record(task):
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


records = []
torch.cuda.reset_peak_memory_stats()
started_all = time.perf_counter()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for index, task in enumerate(tasks, 1):
        started = time.perf_counter()
        try:
            encoded = encode_scoring(as_scoring(raw_record(task), False), tokenizer, 8192)
            row = {
                "input_ids": encoded["prompt_token_ids"],
                "candidate_ids": encoded["candidate_token_ids"],
            }
            batch = {k: v.to("cuda") for k, v in collator([row]).items()}
            logits = candidate_logits(model, batch)[0, : len(encoded["candidate_token_ids"])].float().cpu()
            values = [str(v).lower() if isinstance(v, bool) else str(v)
                      for v in encoded["code_to_choice"].values()]
            if task.question["type"] == "noul":
                values = ["yes" if value == "true" else "no" for value in values]
            probs = dict(zip(values, logits.softmax(-1).tolist()))
            scored = score_task(probs, task)
            record = {
                "task_id": task.id,
                "tier": task_tier[task.id],
                "ok": True,
                "latency_s": time.perf_counter() - started,
                "model": "nimble-gap-9b-v1",
                "probs_source": "candidate_logits",
                "cost_usd": None,
                "cost_basis": "self_hosted_unpriced",
                **scored,
            }
        except Exception as exc:
            record = {
                "task_id": task.id,
                "tier": task_tier[task.id],
                "ok": False,
                "valid": False,
                "strict_valid": False,
                "renormalized": False,
                "correct": False,
                "predicted": None,
                "probs": None,
                "latency_s": time.perf_counter() - started,
                "model": "nimble-gap-9b-v1",
                "probs_source": "candidate_logits",
                "cost_usd": None,
                "cost_basis": "self_hosted_unpriced",
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        if index % 10 == 0 or index == len(tasks):
            print(json.dumps({"progress": f"{index}/{len(tasks)}", "correct": sum(bool(r.get("correct")) for r in records),
                              "failures": sum(not r["ok"] for r in records)}), flush=True)

summary = summarize(tasks, records)
by_tier = {
    tier: summarize([t for t in tasks if task_tier[t.id] == tier],
                    [r for r in records if r["tier"] == tier])
    for tier in tiers
}
payload = {
    "protocol": "JevBench public tasks, upstream Nimble prompt/readout, 8192-token serving limit",
    "nimble_commit": NIMBLE_COMMIT,
    "jevbench_commit": JEVBENCH_COMMIT,
    "dataset_hash": dataset_hash(tasks),
    "adapter_archive_sha256": ARCHIVE_SHA256,
    "model": contract["model"],
    "model_revision": contract["revision"],
    "elapsed_s": time.perf_counter() - started_all,
    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "summary": summary,
    "by_tier": by_tier,
    "records": records,
}
output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(json.dumps({
    "output": str(output),
    "overall": {k: summary[k] for k in ("n_attempted", "n_correct", "accuracy", "operational_success", "brier_mean", "ece", "latency")},
    "by_tier": {tier: {k: report[k] for k in ("n_attempted", "n_correct", "accuracy", "brier_mean", "ece")}
                for tier, report in by_tier.items()},
    "elapsed_s": payload["elapsed_s"],
    "peak_gpu_allocated_gib": payload["peak_gpu_allocated_gib"],
}, indent=2), flush=True)
