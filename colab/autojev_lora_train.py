#!/usr/bin/env python3
"""Train an AutoJev-style Qwen LoRA with a dedicated 255-way readout.

The defaults reproduce the completed 27B experiment. Checkpoints include the
optimizer and RNG state so an interrupted public rerun can resume exactly.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers.masking_utils import create_recurrent_attention_mask

from autojev.evaluate import fit_temperature, hard_label, label_index, metrics, options, read_rows, evaluate_logits
from autojev.model import DecisionModel

MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def enable_noncausal_full_attention(text_model):
    """Lift causality in SDPA layers while preserving padding and recurrence.

    This is the focused architectural change published with
    perplexity-ai/pplx-decider-v1.1-27b. Linear-attention layers keep their
    native recurrent mask.
    """
    if text_model.config._attn_implementation != "sdpa":
        raise ValueError("Noncausal full attention requires SDPA")

    def mask_inputs(module, args, kwargs):
        if args:
            raise ValueError("Noncausal full attention requires keyword inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Noncausal classification does not support a KV cache")
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
            raise ValueError("Expected the processor's 2D padding mask")
        kwargs["attention_mask"] = {
            "full_attention": padding[:, None, None, :].bool(),
            "linear_attention": create_recurrent_attention_mask(
                config=module.config, inputs_embeds=embeddings, attention_mask=padding,
            ),
        }
        return args, kwargs

    text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)


def shuffled(rows, rng):
    result = copy.deepcopy(list(rows))
    for row in result:
        if row["question"]["type"] != "choice":
            continue
        criteria, target = row["question"]["criteria"], row["target"]
        soft = dict(zip(criteria, target, strict=True)) if isinstance(target, list) else None
        items = list(criteria.items())
        rng.shuffle(items)
        row["question"]["criteria"] = dict(items)
        if soft is not None:
            row["target"] = [soft[key] for key, _ in items]
    return result


def target_tensor(rows, device):
    values = torch.zeros((len(rows), 255), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        labels, target = options(row["question"]), row["target"]
        if isinstance(target, list):
            distribution = target
        elif row["question"]["type"] == "noul":
            distribution = [1.0 - float(target), float(target)]
        else:
            distribution = [float(label == target) for label in labels]
        values[i, :len(distribution)] = torch.tensor(distribution, device=device)
    return values


@torch.inference_mode()
def infer(model, rows, batch_size, max_length):
    model.eval()
    all_logits = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch = model.prepare(batch_rows, max_length=max_length)
        logits = model(batch).cpu()
        all_logits.extend(logits[i, :len(options(row["question"]))].tolist() for i, row in enumerate(batch_rows))
    model.train()
    return all_logits


def evaluate(model, dev, calibration, batch_size, max_length):
    cal_logits = infer(model, calibration, batch_size, max_length)
    temperature = fit_temperature(cal_logits, [label_index(options(r["question"]), hard_label(r)) for r in calibration])
    dev_logits = infer(model, dev, batch_size, max_length)
    return temperature, metrics(evaluate_logits(dev, dev_logits, temperature))


def save_artifact(model, out, temperature, config, optimizer=None):
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "adapter")
    save_file({"weight": model.readout.weight.detach().cpu().contiguous()}, out / "readout.safetensors")
    model.processor.save_pretrained(out / "processor")
    config = {**config, "temperature": temperature, "codes": model.codes, "token_ids": model.token_ids}
    (out / "decision_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if optimizer is not None:
        torch.save({
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        }, out / "training_state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/content/autojev-data"))
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--dev-file", default="dev.jsonl")
    ap.add_argument("--temperature-file", default="temperature.jsonl")
    ap.add_argument("--base-model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b/reproduction"))
    ap.add_argument("--steps", type=int, default=1533)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--resume-artifact", type=Path)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--eval-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument(
        "--attention-mode", choices=("causal", "noncausal_full_attention"), default="causal",
        help="Lift the causal mask in SDPA layers as in pplx-decider-v1.1-27b.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    train_path = Path(args.train_file) if Path(args.train_file).is_absolute() else args.data / args.train_file
    dev_path = Path(args.dev_file) if Path(args.dev_file).is_absolute() else args.data / args.dev_file
    temperature_path = (Path(args.temperature_file) if Path(args.temperature_file).is_absolute()
                        else args.data / args.temperature_file)
    train = read_rows(train_path)
    dev = read_rows(dev_path)[:args.eval_rows]
    calibration = read_rows(temperature_path)[:args.eval_rows]
    rng = random.Random(args.seed)
    rng.shuffle(train)

    model = DecisionModel(train=True, base_model=args.base_model, revision=args.revision,
                          gradient_checkpointing=True, cpu_threads=8)
    if args.attention_mode == "noncausal_full_attention":
        enable_noncausal_full_attention(model.backbone.language_model)
    if args.resume_artifact:
        model.backbone = PeftModel.from_pretrained(
            model.backbone, args.resume_artifact / "adapter", is_trainable=True,
        )
        model.readout.load_state_dict(load_file(str(args.resume_artifact / "readout.safetensors")))
    else:
        model.backbone = get_peft_model(model.backbone, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    model.readout.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    config = {
        "format_version": 1, "architecture": "autojev-readout+lora", "base_model": args.base_model,
        "revision": args.revision, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "max_length": args.max_length, "micro_batch": args.micro_batch, "grad_accum": args.grad_accum, "seed": args.seed,
        "attention_mode": args.attention_mode, "pooling": "last",
        "train_rows_available": len(train), "dev_rows": len(dev), "temperature_rows": len(calibration),
        "start_step": args.start_step,
        "resume_artifact": str(args.resume_artifact) if args.resume_artifact else None,
        "trainable_parameters": sum(p.numel() for p in trainable),
    }
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    log = []
    if args.resume_artifact and (args.resume_artifact / "training_state.pt").is_file():
        state = torch.load(args.resume_artifact / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        print(json.dumps({"resumed_optimizer": True, "start_step": args.start_step}), flush=True)
    elif args.resume_artifact:
        print(json.dumps({"resumed_optimizer": False, "warning": "checkpoint has no training_state.pt"}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.start_step + 1, args.steps + 1):
        losses = []
        for micro in range(args.grad_accum):
            first = ((step - 1) * args.grad_accum + micro) * args.micro_batch
            rows = shuffled([train[(first + j) % len(train)] for j in range(args.micro_batch)], rng)
            batch = model.prepare(rows, max_length=args.max_length)
            logits = model(batch)
            target = target_tensor(rows, logits.device)
            loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        entry = {"step": step, "loss": sum(losses) / len(losses), "allocated_gib": torch.cuda.memory_allocated() / 2**30}
        log.append(entry); print(json.dumps(entry), flush=True)
        if args.save_every and step % args.save_every == 0:
            save_artifact(model, args.out / f"checkpoint-{step:05d}", 1.0,
                          {**config, "step": step, "examples_seen": step * args.grad_accum * args.micro_batch},
                          optimizer)
    temperature, dev_metrics = evaluate(model, dev, calibration, 1, args.max_length)
    report = {
        "status": "complete", "steps": args.steps, "examples_seen": args.steps * args.grad_accum * args.micro_batch,
        "temperature": temperature, "development": dev_metrics,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "last_loss": log[-1]["loss"],
    }
    save_artifact(model, args.out / "selected", temperature, {**config, **report}, optimizer)
    (args.out / "training.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
