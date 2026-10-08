# Reproduction guide

## 1. Verify the checkout

```bash
python3 scripts/verify_public_repo.py
```

The verifier checks committed JSON/JSONL syntax, frozen dataset hashes and row counts, notebook cleanliness, file-size policy, and common secret patterns.

## 2. Minimal MLX experiment (0.5B)

Tested on an M1 Pro with 16 GiB unified memory and Python 3.11.

```bash
./setup.sh --small
.venv/bin/python mlx_jev_repro.py --out runs/smoke-repeat
```

What it does:

- downloads the pinned `mlx-community/Qwen2.5-0.5B-Instruct-4bit` revision;
- constructs one shared state/schema prefix with Choice, Score, and Noul branches;
- scores distinct one-token candidate letters without generating output text;
- trains rank-4 query/value adapters in the last four blocks for 12 supervised and four bandit updates;
- fits one scalar temperature on validation only;
- writes config, splits, logits, predictions, adapter, calibration, and report under the requested output directory.

Publication-verification characteristics: 45,056 trainable parameters, 5.23 seconds of training compute, 9.34 seconds after imports end-to-end, 0.551 GB MLX peak allocation, and about 68 ms for a warm three-field call. Initial dependency/model download is additional. Metal timings and tiny-sample metrics vary across OS/MLX versions; compare the regenerated `report.json` with `results/mlx/smoke-0.5b-report.json` rather than expecting byte-identical floats.

## 3. Faithful public Qwen-RLCD path (1.5B)

```bash
./setup.sh --full
.venv/bin/python faithful_rlcd.py --out runs/faithful-repeat
```

This preserves the public engine's prompt, answer-token boundary, shared prefill, suffix batching, and candidate softmax. Training uses rank-4 LoRA on query/value projections in the final four blocks, 80 supervised updates, accumulation 4, learning rate `5e-5`, and CE plus half the multiclass Brier loss. Validation selects the checkpoint and fits temperature; test is untouched until the end.

The exact committed splits are in `data/mlx/faithful-v1`. The reference M1 run took 147 seconds end-to-end and 1.194 GB of MLX-allocated memory.

Direct inference after training:

```bash
.venv/bin/python faithful_rlcd.py \
  --adapter runs/faithful-repeat/best \
  --state "A customer reports unauthorized transfers occurring right now."
```

## 4. Other M1 experiments

```bash
# Nimble recipe scaled to 1.5B (about 63 minutes for the full rank-16 run).
.venv/bin/python nimble_1_5b_mlx.py \
  --rank 16 --layers 28 --out runs/nimble-1.5b-r16-all-repeat

# Banking-only minimal pairs.
.venv/bin/python banking_contrastive_mlx.py \
  --out runs/banking-contrastive-v3-repeat

# Broader banking curriculum.
.venv/bin/python banking_v4_mlx.py \
  --out runs/banking-v4-repeat
```

The Nimble scale-down is a documented negative result: the 1.5B adapter collapsed to candidate position B. Do not report its improved NLL as improved classification; inspect accuracy, pair accuracy, and order-shuffled predictions together.

## 5. Colab 9B experiments

Open `notebooks/autojev_9b_lora.ipynb` for the first AutoJev-style LoRA pilot. The `colab/` directory also includes the later Nimble data-mix trainer and JevBench evaluator. Upstream sources are pinned in `THIRD_PARTY.md`.

The completed Nimble 9B run used Qwen3.5-9B, rank-16 LoRA, 2,676 contrastive examples, effective batch 8, and 335 optimizer steps. Its frozen 324-row holdout rose from 66.4% to 87.7% accuracy. See `results/colab/nimble-9b/`.

## 6. Colab 27B experiment

Open `notebooks/autojev_27b_reproduction.ipynb` and run cells in order. The notebook mounts Drive interactively, installs the pinned AutoJev commit, builds data, filters overlength branches, performs a one-step preflight, and launches the full run.

Reference configuration:

| Parameter | Value |
|---|---|
| Base | `Qwen/Qwen3.8-27B` |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Architecture | AutoJev 255-way readout + rank-16 LoRA |
| Trainable parameters | 80,997,376 |
| Maximum branch length | 512 tokens, strict rejection |
| Retained training rows | 49,039 |
| Microbatch / accumulation | 4 / 8 (effective 32) |
| Optimizer steps | 1,533 |
| Learning rate | `5e-5` |
| Calibration | scalar temperature, separate 512-row fold |
| Reference GPU | RTX PRO 6000 Blackwell, 98 GB |
| Peak allocated memory | 52.43 GiB |
| Final resumed wall time | 11,786 seconds |

The historical job was interrupted and resumed from checkpoint 300. That checkpoint contained adapter/readout weights but no AdamW state, so optimizer moments restarted. The frozen report is the result of that run. The public trainer now saves `training_state.pt` with AdamW and RNG state; future resumes are exact.

## 7. Live Jev comparison

Create `.env` locally or add `TYPESAFE_API_KEY` to Colab Secrets. Never paste the key into source or a notebook cell.

```bash
set -a
. ./.env
set +a
python colab/eval_autojev27b_vs_jev.py \
  --artifact /path/to/selected \
  --eval-file data/jev_gap_curriculum_v1/eval.jsonl \
  --output runs/autojev27b-vs-jev \
  --limit 128 --batch-size 4 --workers 4 --jev-model jev-1.13.0
```

The evaluator caches completed Jev responses, records normalized candidate probabilities and input hashes, and joins reference labels only after inference. The committed three-way report used identical 128 case IDs for public AutoJev-27B, our banking finetune, and Jev.

## 8. Scaling safely

- Increase model size only after the one-step preflight saves and reloads a checkpoint.
- Keep the calibration fold and final test isolated by family, not just row.
- Preserve option-order augmentation and report shuffled-order results.
- Prefer adding diverse, reviewed counterfactual families over repeating templates.
- Record model/data revisions, dataset hashes, per-case predictions, optimizer state, and RNG state.
- Put checkpoints on Drive or a model registry. Keep only compact configs/reports in Git.
