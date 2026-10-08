# Reproduction guide

Run this first after cloning:

```bash
python3 scripts/verify_public_repo.py
```

It verifies committed JSON/JSONL files, dataset hashes, notebook cleanliness, file-size limits, the publication manifest, and common secret patterns.

## 1. Basic MLX recipe

Tested on an M1 Pro with 16 GiB unified memory and Python 3.11.

```bash
./setup.sh --small
.venv/bin/python mlx_jev_repro.py --out runs/mlx-smoke
```

The script constructs a shared state/schema prefix with `choice`, `score`, and `noul` branches; scores one-token candidates without generating output text; trains rank-4 query/value LoRA adapters in the last four blocks; runs 12 supervised plus four bandit-style updates; and fits a scalar temperature on validation data only.

It writes the configuration, splits, logits, predictions, adapter, calibration data, and report beneath `runs/mlx-smoke`. The frozen reference is `results/mlx/smoke-0.5b-report.json`.

Inference with the saved adapter:

```bash
.venv/bin/python mlx_jev_repro.py \
  --adapter runs/mlx-smoke/adapter \
  --state "A customer reports an unauthorized transfer in progress."
```

Add `--compare-jev` after exporting `TYPESAFE_API_KEY` to score the same input with Jev.

## 2. Nimble scale-down on MLX

```bash
./setup.sh --full
.venv/bin/python nimble_1_5b_mlx.py \
  --rank 16 \
  --layers 28 \
  --out runs/nimble-1.5b
```

This uses the pinned Nimble repository's public data and prompt code. The full reference run took about 63 minutes on an M1 Pro. It collapsed toward candidate position B, so treat it as a documented failure mode rather than a successful small-model result. Inspect normal and option-shuffled accuracy alongside NLL and Brier score. The frozen report is `results/mlx/nimble-1.5b-report.json`.

For a quick code-path check:

```bash
.venv/bin/python nimble_1_5b_mlx.py \
  --max-steps 2 --eval-limit 16 --skip-shuffled-eval \
  --out runs/nimble-smoke
```

## 3. Nimble 9B Colab run

In a fresh GPU Colab notebook, mount Drive and run:

```bash
!git clone https://github.com/jsaurabh/qwen-rlcd-jev-repro.git /content/qwen-rlcd-jev
!git clone https://github.com/bespokelabsai/nimble.git /content/nimble
!git -C /content/nimble checkout --detach f136b3f75721fda4ea961f73993cc50b08488835
!python -m pip install -q -e /content/nimble
```

Then launch the data-mix run. Its checkpoints go directly to Drive:

The later data-mix experiment uses:

```bash
!python -u /content/qwen-rlcd-jev/colab/train_gap_mix.py \
  --nimble /content/nimble \
  --gap /content/qwen-rlcd-jev/data/jev_gap_curriculum_v1 \
  --out /content/drive/MyDrive/qwen-rlcd-jev/nimble-gap-9b
```

Generate the committed gap curriculum locally with:

```bash
python3 tools/generate_gap_curriculum.py \
  --output data/jev_gap_curriculum_v1 \
  --pairs-per-family 300 \
  --seed 20260921
```

Frozen reports are under `results/colab/nimble-9b/` and `results/colab/nimble-data-mix-v2/`.

## 4. AutoJev-style 27B Colab run

Open `notebooks/autojev_9b_lora.ipynb` for the lower-cost preflight or `notebooks/autojev_27b_reproduction.ipynb` for the completed large experiment. The 27B notebook mounts Google Drive, installs the pinned AutoJev source, builds and filters the data, performs a one-step preflight, and launches the full trainer.

Reference configuration:

| Parameter | Value |
|---|---|
| Base | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Trainable layers | rank-16 LoRA plus 255-way readout |
| Trainable parameters | 80,997,376 |
| Maximum branch length | 512 tokens, strict rejection |
| Training rows | 49,039 |
| Microbatch / accumulation | 4 / 8 |
| Optimizer steps | 1,533 |
| Learning rate | `5e-5` |
| Calibration | scalar temperature on a separate fold |
| Reference GPU | RTX PRO 6000 Blackwell, 98 GB |
| Peak allocated memory | 52.43 GiB |

The historical run resumed after an interruption from a checkpoint that did not contain optimizer state. The public trainer now saves LoRA/readout weights, AdamW state, and RNG state so future resumes are exact.

## 5. Compare the 27B model with Jev

Put `TYPESAFE_API_KEY` in Colab Secrets or export it in the shell. Do not paste it into a notebook cell.

```bash
python -u colab/eval_autojev27b_vs_jev.py \
  --artifact /path/to/selected \
  --eval-file data/jev_gap_curriculum_v1/eval.jsonl \
  --output runs/autojev27b-vs-jev \
  --limit 128 \
  --batch-size 4 \
  --workers 4 \
  --jev-model jev-1.13.0
```

The evaluator caches completed API responses, records normalized candidate probabilities and input hashes, and joins labels only after inference. To evaluate the public model without the banking finetune, use `colab/eval_public_autojev27b.py`.

## Practical scaling rules

- Run a one-step preflight before committing to a large job.
- Save checkpoints to Drive, including optimizer and RNG state.
- Split calibration and test sets by problem family, not only by row.
- Randomize option order during training and report shuffled-order accuracy.
- Record immutable model/data revisions, hashes, and per-case predictions.
- Keep weights and secrets out of Git; commit only compact evidence.
