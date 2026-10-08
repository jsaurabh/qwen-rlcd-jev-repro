# Jev-like decision models: three reproducible recipes

[![MLX](https://img.shields.io/badge/Apple_Silicon-MLX-black)](REPRODUCING.md#1-basic-mlx-recipe)
[![Open 9B run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jsaurabh/qwen-rlcd-jev-repro/blob/main/notebooks/autojev_9b_lora.ipynb)
[![Open 27B run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jsaurabh/qwen-rlcd-jev-repro/blob/main/notebooks/autojev_27b_reproduction.ipynb)

This repository publishes the three experiments used in our Jev talk:

| Recipe | Model and hardware | What it demonstrates |
|---|---|---|
| **Basic MLX** | Qwen2.5-0.5B 4-bit on an M1 Pro | Typed `choice`, `score`, and `noul` decisions; candidate-token scoring; LoRA; temperature calibration |
| **Nimble** | Qwen2.5-1.5B on MLX and Qwen3.5-9B on Colab | Contrastive candidate classification, option-order controls, a targeted data mix, and public JevBench evaluation |
| **AutoJev-style Colab** | Qwen3.8-27B on a 98 GB RTX Pro 6000 | A 255-way learned readout plus LoRA, strict length filtering, checkpoint/resume, calibration, and comparison with public AutoJev and Jev |
| **Perplexity decider v1.1** | Released Qwen3.8-27B checkpoint on a 98 GB RTX Pro 6000 | Noncausal full-attention layers, exact released inference, and a controlled path from our LoRA recipe toward the 61.56 Decision Index model |

The common idea is to turn a decoder model into a decision model: serialize state and a typed question, score only valid candidates instead of generating an answer, train those scores with classification objectives, then calibrate them on held-out data.

This is an independent reconstruction from public work. It does **not** reproduce TypeSafe's undisclosed Jev architecture, weights, or training data.

## Talk deck

The accompanying 21-slide technical overview is available at [`slides/jev_decision_models_balanced_technical_talk_v8.pptx`](slides/jev_decision_models_balanced_technical_talk_v8.pptx). It covers the Jev execution model, open architecture families, training and calibration, and the public benchmark results summarized below.

## Start with the MLX recipe

On an Apple-Silicon Mac with Python 3.11:

```bash
git clone https://github.com/jsaurabh/qwen-rlcd-jev-repro.git
cd qwen-rlcd-jev-repro
./setup.sh --small
.venv/bin/python mlx_jev_repro.py --out runs/mlx-smoke
```

This run is intentionally small. Our M1 Pro verification trained 45,056 parameters for 16 updates in 5.23 seconds and peaked at 0.551 GB of MLX-allocated memory. The purpose is to make the complete mechanics easy to inspect; it is not a competitive benchmark result.

## Reproduce the Nimble experiments

The local scale-down uses Nimble's public candidate-classification format and controls:

```bash
./setup.sh --full
.venv/bin/python nimble_1_5b_mlx.py \
  --rank 16 --layers 28 \
  --out runs/nimble-1.5b
```

That M1 run is a useful negative result: the 1.5B adapter collapsed toward one answer position. Its report is retained because it shows why accuracy, option-order tests, and calibration must be read together.

The Colab path scales the same family of ideas to Qwen3.5-9B and mixes Nimble data with the committed contrastive gap curriculum. Follow the exact setup and training commands in [REPRODUCING.md](REPRODUCING.md#3-nimble-9b-colab-run).

## Reproduce the 27B Colab run

The 9B notebook is the inexpensive preflight; the 27B notebook reproduces the full run. Both use the same LoRA-plus-readout recipe. They build the text-only AutoJev corpus, perform a memory preflight, and write checkpoints to Google Drive. The 27B path also applies the committed gap mix and strict 512-token filtering.

The completed run used rank-16 LoRA plus a 255-way readout, 49,039 training rows, an effective batch of 32, and 1,533 optimizer steps. It peaked at 52.43 GiB of allocated GPU memory.

## Call your model through an API

Use the lightweight Python client and local GPU server in [`decision_api/`](decision_api/).
The primary interface is `/v1/decisions`, with `/v1/systemone` compatibility for
existing Jev experiments. Both return full candidate distributions from the same
saved model. See [API comparison, setup, and examples](docs/decision-api.md).

```bash
python -m examples.banking_decisions --provider local
```

Start the server first using the linked setup guide. The same example can call
OpenAI Decisions or Jev with `--provider openai` or `--provider jev`.

## Frozen results

These numbers describe the committed protocols and datasets. They are not general model rankings.

| Run | Evaluation | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| Nimble 9B | 324-row holdout | 87.7% | 0.458 | 0.210 | — |
| Nimble + gap data mix | Public JevBench subset, 231 rows | 79.7% | — | 0.290 | 0.089 |
| Our 27B finetune | Same 128 banking cases | 91.4% | 0.153 | 0.107 | 0.021 |
| Perplexity decider v1.1 27B | Same 128 banking cases | 92.2% | 0.159 | 0.107 | 0.053 |
| Public AutoJev-27B | Same 128 banking cases | 89.1% | 0.235 | 0.142 | 0.082 |
| Jev 1.13.0 | Same 128 banking cases | 98.4% | 0.090 | 0.039 | 0.062 |

The 128-case evaluation is narrow, synthetic, and banking-focused. Per-case outputs are in [`results/colab/autojev-27b-v4/three-way-comparison.json`](results/colab/autojev-27b-v4/three-way-comparison.json).

See [`docs/pplx-decider-v1.1.md`](docs/pplx-decider-v1.1.md) for the released
training configuration, the exact-checkpoint result, the limits of a
single-GPU reproduction, and the noncausal LoRA ablation command.

## Repository map

```text
mlx_jev_repro.py                 basic MLX recipe
nimble_1_5b_mlx.py              Nimble scale-down on Apple Silicon
nimble_public_eval_mlx.py       public-data evaluator for the MLX adapter
data/jev_gap_curriculum_v1/     deterministic contrastive curriculum
colab/                          shared CUDA trainers and evaluators
notebooks/                      9B and 27B Colab entry points
results/                        compact reports from completed runs
```

Model weights, adapters, optimizer state, API keys, and notebook output are intentionally excluded. Live Jev evaluation reads `TYPESAFE_API_KEY` from the environment or Colab Secrets.

See [REPRODUCING.md](REPRODUCING.md) for exact commands and [THIRD_PARTY.md](THIRD_PARTY.md) for pinned upstream revisions.

## License

Original code and synthetic data in this repository are MIT licensed. Third-party code, models, and benchmarks retain their own licenses.
