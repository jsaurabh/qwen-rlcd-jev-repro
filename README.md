# Reproducing Jev-like decision models with Qwen

This repository contains the code, synthetic data, frozen reports, and Colab workflows used in our Jev-like decision-model experiments. It is designed so a technically experienced reader can reproduce the Apple-Silicon MLX runs, inspect every evaluation input, and rerun the larger CUDA experiments.

The central pattern is simple:

1. serialize application state plus a typed question (`choice`, `noul`, or `score`);
2. prefill the shared state/schema once where the inference engine supports it;
3. score only valid candidate tokens or a learned candidate readout instead of generating JSON;
4. train on decision labels, including contrastive counterfactuals and option-order augmentation;
5. fit temperature on a separate calibration split and report accuracy, NLL, Brier, ECE, latency, and order sensitivity.

This is an independent reconstruction using public Qwen, Qwen-RLCD, Nimble, AutoJev, and JevBench material. It does **not** reproduce TypeSafe's undisclosed Jev architecture or proprietary training data.

## What is included

| Track | Hardware/model | Entry point | Frozen evidence |
|---|---|---|---|
| Minimal typed decisions | M1 Pro, Qwen2.5-0.5B 4-bit | `mlx_jev_repro.py` | `results/mlx/smoke-0.5b-report.json` |
| Public Qwen-RLCD parity + LoRA | M1 Pro, Qwen2.5-1.5B 4-bit | `faithful_rlcd.py` | `results/mlx/faithful-1.5b-report.json` |
| Banking curricula | M1 Pro, Qwen2.5-1.5B 4-bit | `banking_contrastive_mlx.py`, `banking_v4_mlx.py` | `data/mlx/`, `results/mlx/` |
| Nimble scale-down | M1 Pro, Qwen2.5-1.5B 4-bit | `nimble_1_5b_mlx.py` | `results/mlx/nimble-1.5b-report.json` |
| Nimble/AutoJev CUDA runs | Colab, Qwen3.5-9B | `notebooks/autojev_9b_lora.ipynb`, `colab/` | `results/colab/nimble-*` |
| AutoJev-style large run | RTX Pro 6000 Blackwell, Qwen3.8-27B | `notebooks/autojev_27b_reproduction.ipynb` | `results/colab/autojev-27b-v4/` |

All committed data is synthetic. It contains no customer records. Frozen result files include predictions and metrics, not secrets or model weights.

## Fastest reproduction on Apple Silicon

Requirements: an Apple-Silicon Mac, Python 3.11, Git, and roughly 1 GB of free space for the 0.5B smoke test.

```bash
git clone https://github.com/jsaurabh/qwen-rlcd-jev-repro.git
cd qwen-rlcd-jev-repro
./setup.sh
.venv/bin/python mlx_jev_repro.py --out runs/smoke-repeat
```

The publication-verification M1 Pro run trained 45,056 LoRA parameters for 16 updates, completed the training loop in 5.23 seconds, and used 0.551 GB of MLX-allocated memory. It is a plumbing/calibration smoke test, not a competitive model. The earlier pre-banking-schema report is preserved separately as `results/mlx/smoke-0.5b-legacy-report.json`.

To run the faithful 1.5B experiment, fetch the larger model and references first:

```bash
./setup.sh --full
.venv/bin/python faithful_rlcd.py --out runs/faithful-repeat
```

The reference run used 80 updates and finished end-to-end in 147 seconds on an M1 Pro. Exact frozen datasets are under `data/mlx/faithful-v1`; the script regenerates and hashes them before evaluation.

## Colab reproduction

Open [`notebooks/autojev_27b_reproduction.ipynb`](notebooks/autojev_27b_reproduction.ipynb) in Colab for the large run. The completed experiment used:

- `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- rank-16 LoRA plus a fully trained 255-way readout (80,997,376 trainable parameters);
- 49,039 retained training rows after a strict 512-token filter;
- microbatch 4, accumulation 8, 1,533 optimizer steps;
- 52.43 GiB peak allocated GPU memory and 11,786 seconds for the final resumed segment.

The notebook writes checkpoints to Google Drive and includes restart-safe optimizer/RNG state. A GPU with at least 64 GB of usable memory is recommended for these exact settings. Use `--micro-batch 1` for a memory preflight. The 9B notebook is substantially cheaper and is the recommended first CUDA reproduction.

## Main measured results

These are experiment results on synthetic or public benchmark data, not production claims.

| Experiment | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| M1 faithful Qwen-RLCD base, 180 decisions | 76.1% | 0.707 | 0.378 | 0.082 |
| M1 faithful Qwen-RLCD + selected LoRA | 82.8% | 0.496 | 0.280 | 0.083 |
| Colab Nimble 9B holdout, 324 decisions | 87.7% | 0.458 | 0.210 | — |
| Colab 27B banking finetune, 128 cases | 91.4% | 0.153 | 0.107 | 0.021 |
| Public AutoJev-27B, same 128 cases | 89.1% | 0.235 | 0.142 | 0.082 |
| Jev 1.13.0, same 128 cases | 98.4% | 0.090 | 0.039 | 0.062 |

The 128-case comparison used identical held-out inputs and joined references only after inference. The dataset is narrow, synthetic, and banking-focused; it does not establish general superiority. See `results/colab/autojev-27b-v4/three-way-comparison.json` for every row.

## Repository map

- `mlx_jev_repro.py`: compact state/schema prefill, three typed decisions, candidate scoring, soft-label training, RLCD-style bandit smoke stage, and temperature fitting.
- `faithful_rlcd.py`: parity-first use of the original public Qwen-RLCD engine with a separately trained LoRA.
- `banking_*`, `nimble_1_5b_mlx.py`: the subsequent MLX data/training experiments.
- `data/`: committed frozen synthetic splits plus deterministic generators.
- `colab/`: CUDA dataset preparation, trainers, and Jev/AutoJev evaluators.
- `notebooks/`: clean Colab entry points with no saved credentials or output.
- `results/`: compact reports and per-case benchmark records; checkpoints are intentionally excluded.
- `REPRODUCING.md`: exact experiment commands, expected resources, and provenance checks.
- `THIRD_PARTY.md`: upstream URLs, roles, and immutable revisions.

## Secrets and weights

Live Jev evaluation reads `TYPESAFE_API_KEY` from the environment or Colab Secrets. Copy `.env.example` to `.env` locally. `.env`, tokens, downloaded bases, LoRA checkpoints, optimizer states, logs, and Drive mounts are ignored by Git. Never put an API key in a notebook cell.

## License

Original code and synthetic data in this repository are MIT licensed. Third-party source, models, and benchmarks retain their own licenses; see `THIRD_PARTY.md`.
