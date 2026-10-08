#!/usr/bin/env python3
"""Provision source, build the public text corpus, and run the 9B preflight."""
import subprocess
from pathlib import Path

def run(*args):
    print("+", *args, flush=True)
    subprocess.run(args, check=True)

run("git", "clone", "--depth", "1", "https://github.com/denis-pplx/autojev.git", "/content/autojev")
run("python", "-m", "pip", "install", "-q", "-e", "/content/autojev", "peft>=0.17", "datasets>=3.0")
data_py = Path("/content/autojev/src/autojev/data.py")
source = data_py.read_text()
source = source.replace(
    '"maximum_input_tokens": max(cast(int, row["source"]["input_tokens"]) for row in rows),',
    '"maximum_input_tokens": max((cast(int, row["source"]["input_tokens"]) for row in rows), default=0),',
)
data_py.write_text(source)
if not Path("/content/autojev-data/train.jsonl").exists():
    run("autojev-data", "--output", "/content/autojev-data", "--manifest", "/content/autojev-data-manifest.json",
        "--train-scale", "0.5", "--image-train", "0", "--image-eval", "0", "--max-length", "8192")
run("python", "/content/autojev_9b_lora_train.py", "--steps", "20", "--grad-accum", "8", "--eval-rows", "64")
