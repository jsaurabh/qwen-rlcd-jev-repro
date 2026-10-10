#!/usr/bin/env python3
"""Train an AutoJev-style Qwen LoRA with a dedicated 255-way readout.

Use the explicit pilot or full-epoch commands in README.md. Checkpoints include
optimizer and RNG state for resume; initialization alone resets the optimizer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from objective import decision_rl_loss
from reference_cache import reference_cache, artifact_hash
from token_plan import make_plan, updates, learning_rate

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
        if padding.shape != embeddings.shape[:2] or not padding.bool().any(dim=-1).all():
            raise ValueError("Padding mask must match complete nonempty inputs")
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


def save_artifact(model, out, temperature, config, optimizer=None, sampler_rng=None, drive_store=None):
    destination = out
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    out = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    out.mkdir(parents=True, exist_ok=False)
    import shutil
    for name in ["run_config.json", "training.jsonl", "evaluation.jsonl"]:
        if (destination.parent / name).exists():
            shutil.copyfile(destination.parent / name, out / name)
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
            "sampler_rng": sampler_rng.getstate() if sampler_rng else None,
            "step": config.get("step", config.get("steps", 0)),
        }, out / "training_state.pt")
    (out / "COMPLETE.json").write_text(json.dumps({"step": config.get("step", config.get("steps", 0))}))
    out.rename(destination)
    print(json.dumps({"checkpoint_saved": str(destination)}), flush=True)
    if drive_store is not None:
        receipt = drive_store.upload_checkpoint(destination)
        receipt_path = destination.with_name(destination.name + ".drive-receipt.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"checkpoint_verified_on_drive": str(destination), **receipt}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-artifact',type=Path,required=True)
    ap.add_argument('--reference-cache',type=Path,required=True)
    ap.add_argument('--objective',choices=['sft','rl_hybrid'],required=True)
    ap.add_argument('--beta',type=float,default=.05)
    ap.add_argument('--ce-weight',type=float,default=.25)
    ap.add_argument("--data", type=Path, default=Path("/content/autojev-data"))
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--dev-file", default="dev.jsonl")
    ap.add_argument("--temperature-file", default="temperature.jsonl")
    ap.add_argument("--base-model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/qwen-rlcd-jev/autojev-27b/reproduction"))
    ap.add_argument("--checkpoint-spool", type=Path)
    ap.add_argument("--full-epoch", action="store_true", help="Use each eligible training example exactly once")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--token-budget", type=int, default=2000000)
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
    ap.add_argument("--require-drive", action="store_true")
    ap.add_argument("--drive-folder-id", help="Drive API parent folder; checkpoint uploads are synchronous and verified")
    ap.add_argument("--drive-credentials", type=Path, default=os.environ.get("DRIVE_CHECKPOINT_CREDENTIALS"))
    ap.add_argument(
        "--attention-mode", choices=("causal", "noncausal_full_attention"), default="causal",
        help="Lift the causal mask in SDPA layers as in pplx-decider-v1.1-27b.",
    )
    args = ap.parse_args()
    drive_store = None
    if args.drive_folder_id:
        if args.require_drive:
            ap.error("Use either mounted Drive or API checkpoint uploads")
        if not args.drive_credentials:
            ap.error("Drive API uploads require --drive-credentials or DRIVE_CHECKPOINT_CREDENTIALS")
        from drive_checkpoint_store import DriveStore
        parent_store = DriveStore(args.drive_credentials, args.drive_folder_id)
        drive_store = DriveStore(args.drive_credentials, parent_store.folder(args.out.name))
    if args.checkpoint_spool:
        if drive_store is not None or args.require_drive:
            ap.error("Use one checkpoint transport")
        from checkpoint_mailbox import CheckpointMailbox
        drive_store = CheckpointMailbox(args.checkpoint_spool)
    def check_storage():
        if args.require_drive:
            if not os.path.ismount("/content/drive") or not Path("/content/drive/MyDrive").is_dir():
                raise RuntimeError("Google Drive must be mounted before training/checkpoint writes")
            if not args.out.resolve().is_relative_to(Path("/content/drive/MyDrive").resolve()):
                raise ValueError("--require-drive requires an output directory on Google Drive")
    check_storage()
    if args.start_step < 0 or min(args.micro_batch, args.grad_accum, args.eval_rows) < 1:
        raise ValueError("Require positive batch/eval sizes and nonnegative start-step")
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    train_path = Path(args.train_file) if Path(args.train_file).is_absolute() else args.data / args.train_file
    dev_path = Path(args.dev_file) if Path(args.dev_file).is_absolute() else args.data / args.dev_file
    temperature_path = (Path(args.temperature_file) if Path(args.temperature_file).is_absolute()
                        else args.data / args.temperature_file)
    train = read_rows(train_path)
    dev = read_rows(dev_path)[:args.eval_rows]
    calibration = read_rows(temperature_path)[:args.eval_rows]
    data_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
                   [("train", train_path), ("development", dev_path), ("temperature", temperature_path)]}
    if not train or not dev or not calibration:
        raise ValueError("Train, development, and calibration folds must be nonempty")
    if args.resume_artifact:
        saved_config = json.loads((args.resume_artifact / "decision_config.json").read_text())
        for key, expected in {"attention_mode": args.attention_mode, "base_model": args.base_model,
                              "revision": args.revision, "micro_batch": args.micro_batch,
                              "grad_accum": args.grad_accum, "seed": args.seed,
                              "max_length": args.max_length}.items():
            if saved_config.get(key, "causal" if key == "attention_mode" else None) != expected:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if saved_config.get("data_sha256", data_hashes) != data_hashes:
            raise ValueError("Resume data hashes differ")
    init_config=json.loads((args.init_artifact/'decision_config.json').read_text())
    for key,expected in {'base_model':args.base_model,'revision':args.revision,'attention_mode':args.attention_mode}.items():
        if init_config.get(key)!=expected:raise ValueError(f'Initial artifact mismatch: {key}')
    initial_hash=artifact_hash(args.init_artifact)
    if args.resume_artifact:
        for key,value in {'initial_artifact_sha256':initial_hash,'objective':args.objective,'beta':args.beta,'ce_weight':args.ce_weight}.items():
            if saved_config.get(key)!=value:raise ValueError(f'Resume mismatch: {key}')
    if args.full_epoch:
        args.token_budget = sum(row["input_token_count"] for row in train)
    rng = random.Random(args.seed)
    plan, planned_tokens = make_plan(train, args.token_budget, args.seed)
    update_plan = list(updates(plan, args.micro_batch, args.grad_accum))
    args.steps = len(update_plan)
    if args.start_step >= args.steps:
        raise ValueError("Already at or beyond the final update")
    plan_hash = hashlib.sha256(json.dumps([train[i]["id"] for i in plan]).encode()).hexdigest()
    cursor = sum(len(b) for g in update_plan[:args.start_step] for b in g)
    tokens_seen = sum(train[i]["input_token_count"] for i in plan[:cursor])
    if args.resume_artifact:
        for key, value in {"plan_sha256": plan_hash, "token_budget": args.token_budget,
                           "examples_seen": cursor, "tokens_seen": tokens_seen, "lr": args.lr}.items():
            if saved_config.get(key) != value:
                raise ValueError(f"Resume mismatch: {key}")
        if not (args.resume_artifact / "training_state.pt").is_file():
            raise ValueError("Exact resume requires optimizer and RNG state")

    model = DecisionModel(train=True, base_model=args.base_model, revision=args.revision,
                          gradient_checkpointing=True, cpu_threads=8)
    if args.attention_mode == "noncausal_full_attention":
        enable_noncausal_full_attention(model.backbone.language_model)
    load_from=args.resume_artifact or args.init_artifact
    model.backbone=PeftModel.from_pretrained(model.backbone,load_from/'adapter',is_trainable=True)
    model.readout.load_state_dict(load_file(str(load_from/'readout.safetensors')))
    if model.codes!=init_config['codes'] or model.token_ids!=init_config['token_ids']:
        raise ValueError('Initial artifact answer vocabulary mismatch')
    reference,reference_meta=reference_cache(model,train,plan,args.reference_cache,
        {'plan_sha256':plan_hash,'data_sha256':data_hashes['train'],'initial_artifact_sha256':initial_hash,
         'max_length':args.max_length},args.micro_batch,allow_create=not bool(args.resume_artifact))
    if drive_store is not None and not args.resume_artifact:
        import shutil
        snapshot=args.out/'reference-checkpoint'
        snapshot.mkdir(exist_ok=False)
        shutil.copyfile(args.reference_cache,snapshot/'reference.pt')
        shutil.copyfile(args.reference_cache.with_suffix('.json'),snapshot/'reference.json')
        (snapshot/'COMPLETE.json').write_text(json.dumps({'kind':'frozen-reference','initial_artifact_sha256':initial_hash}))
        receipt=drive_store.upload_checkpoint(snapshot)
        print(json.dumps({'reference_verified_on_drive':receipt}),flush=True)
    reference_positions={idx:pos for pos,idx in enumerate(plan)}
    # Inference/cache creation must not change the training RNG between arms.
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model.readout.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    config = {
        "initial_artifact_sha256":initial_hash,"objective":args.objective,"beta":args.beta,"ce_weight":args.ce_weight,"reference_cache":reference_meta,
        "format_version": 1, "architecture": "autojev-readout+lora", "base_model": args.base_model,
        "revision": args.revision, "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "max_length": args.max_length, "micro_batch": args.micro_batch, "grad_accum": args.grad_accum, "seed": args.seed,
        "attention_mode": args.attention_mode, "pooling": "last",
        "data_sha256": data_hashes, "lr": args.lr, "total_steps": args.steps,
        "token_budget": args.token_budget, "planned_tokens": planned_tokens, "plan_sha256": plan_hash,
        "full_epoch": args.full_epoch, "eval_every": args.eval_every,
        "sampling": "single seeded shuffle; fixed option order; stop before token budget",
        "lr_schedule": "5% token warmup then linear decay",
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
        if state.get("step", args.start_step) != args.start_step:
            raise ValueError("--start-step differs from checkpoint optimizer step")
        if state.get("sampler_rng") is not None:
            rng.setstate(state["sampler_rng"])
        else:
            # Replay option-order draws for old checkpoints that did not save
            # the dedicated random.Random instance used by the sampler.
            for old_micro in range(args.start_step * args.grad_accum):
                first = old_micro * args.micro_batch
                shuffled([train[(first + j) % len(train)] for j in range(args.micro_batch)], rng)
        print(json.dumps({"resumed_optimizer": True, "start_step": args.start_step}), flush=True)
    elif args.resume_artifact:
        print(json.dumps({"resumed_optimizer": False, "warning": "checkpoint has no training_state.pt"}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.start_step + 1, args.steps + 1):
        check_storage()
        step_started = time.monotonic()
        group = update_plan[step - 1]
        decisions = sum(len(indices) for indices in group)
        update_tokens = sum(train[i]["input_token_count"] for indices in group for i in indices)
        lr = learning_rate(tokens_seen + update_tokens / 2, args.token_budget, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        weighted_loss = 0.0
        update_stats={"reward":0.0,"kl":0.0,"ce":0.0}
        for indices in group:
            rows = [train[i] for i in indices]
            batch = model.prepare(rows, max_length=args.max_length)
            observed = batch.inputs["attention_mask"].sum(dim=1).tolist()
            if observed != [r["input_token_count"] for r in rows]:
                raise ValueError("Tokenizer/processor mismatch: precomputed token budget invalid")
            logits = model(batch)
            target = target_tensor(rows, logits.device)
            ref=reference[[reference_positions[i] for i in indices]].to(logits.device)
            rl_loss,stats=decision_rl_loss(logits,target,ref,[len(options(r['question'])) for r in rows],args.beta,args.ce_weight)
            loss=rl_loss if args.objective=='rl_hybrid' else -(target*F.log_softmax(logits,dim=-1)).sum(-1).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            for key,value in stats.items():update_stats[key]+=float(value)*len(rows)/decisions
            weight = len(rows) / decisions
            (loss * weight).backward()
            weighted_loss += float(loss.detach()) * weight
        cursor += decisions
        tokens_seen += update_tokens
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        elapsed = time.monotonic() - step_started
        entry = {**update_stats,"step": step, "loss": weighted_loss, "tokens_seen": tokens_seen, "examples_seen": cursor, "lr": lr, "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                 "step_seconds": elapsed, "examples_per_second": decisions / elapsed, "tokens_per_second": update_tokens / elapsed}
        log.append(entry); print(json.dumps(entry), flush=True)
        with (args.out / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            # Preserve training randomness across diagnostic evaluation.
            py_state = random.getstate()
            try:
                with torch.random.fork_rng():
                    temp, diagnostic = evaluate(model, dev, calibration, args.micro_batch, args.max_length)
            finally:
                random.setstate(py_state)
                model.train()
            milestone = {"step": step, "tokens_seen": tokens_seen, "temperature": temp, "development": diagnostic}
            with (args.out / "evaluation.jsonl").open("a") as stream:
                stream.write(json.dumps(milestone) + "\n")
            print(json.dumps({"phase": "evaluation", "step": step, "temperature": temp,
                **{key: diagnostic[key] for key in ["accuracy", "nll", "brier", "ece"]}}), flush=True)
        if args.save_every and (step == 1 or step % args.save_every == 0 or step == args.steps):
            check_storage()
            save_artifact(model, args.out / f"checkpoint-{step:05d}", 1.0,
                          {**config, "step": step, "examples_seen": cursor, "tokens_seen": tokens_seen},
                          optimizer, rng, drive_store)
    temperature, dev_metrics = evaluate(model, dev, calibration, args.micro_batch, args.max_length)
    dev_logits = infer(model, dev, args.micro_batch, args.max_length)
    suites = sorted({r.get("suite", "unknown") for r in dev})
    by_suite = {}
    for suite in suites:
        indices = [i for i, r in enumerate(dev) if r.get("suite", "unknown") == suite]
        subset = [dev[i] for i in indices]
        logits_subset = [dev_logits[i] for i in indices]
        by_suite[suite] = {"count": len(indices),
            "raw": metrics(evaluate_logits(subset, logits_subset, 1.0)),
            "calibrated": metrics(evaluate_logits(subset, logits_subset, temperature))}

    report = {
        "status": "complete", "step": args.steps, "steps": args.steps, "examples_seen": cursor, "tokens_seen": tokens_seen,
        "temperature": temperature, "development": dev_metrics, "by_suite": by_suite,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "last_loss": log[-1]["loss"],
    }
    check_storage()
    save_artifact(model, args.out / "selected", temperature, {**config, **report}, optimizer, rng, drive_store)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if drive_store is not None and not args.checkpoint_spool:
        metadata_folder = drive_store.folder("run-metadata")
        for name in ("run_config.json", "report.json", "training.jsonl"):
            drive_store.upload_file(args.out / name, metadata_folder)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
