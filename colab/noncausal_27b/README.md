---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: peft
tags:
- decision-model
- lora
- classification
- autojev
---
# Qwen Decision 27B — noncausal LoRA

An independent AutoJev-style experiment: a Qwen 27B backbone, LoRA adapter and 255-way decision readout. This is **not the official Perplexity Decider checkpoint or a reproduction of proprietary Jev**. The model returns probabilities for typed `choice` and `noul` decisions without generating an answer string token by token.

## Public evaluation

Decision Index **0.3 public index: 57.70** (raw index 68.26), complete over 140,178 scoreable requests: 140,148 supported, 30 unsupported, zero execution errors. All unsupported requests count under the official scoring policy. The runner processed 140,620 requests including 442 officially excluded requests.

| Area | Chance-adjusted score |
|---|---:|
| Knowledge & Reasoning | 42.77 |
| Language Understanding | 62.34 |
| Retrieval & Classification | 57.81 |
| Tools & Automation | 80.16 |
| Arts & Human Taste | 43.06 |

[Untouched results and reproduction metadata](https://huggingface.co/datasets/jsaurabh/qwen-decision-27b-decision-index-0.3).
This is a **public-only score**, not an official Full/private leaderboard score. Maintainers must independently verify answers, private performance and latency eligibility.

Measured HTTP latency on one RTX PRO 6000 Blackwell 96 GB: median 105.9 ms, mean 383.2 ms, p95 1565.3 ms. Startup excluded. These numbers are not the maintainers' held-out latency gate. The run used the correct PyTorch causal-convolution fallback, without `causal_conv1d` installed.

## Run the evaluated inference path

Linux, Python 3.13 and a CUDA GPU with approximately 96 GB VRAM are recommended to reproduce the evaluated environment. The base model is downloaded separately; allow ample disk space for its weights. This repository contains only the adapter/readout, processor and code.

```bash
hf download jsaurabh/qwen-decision-27b-noncausal-lora --local-dir qwen-decision-27b
cd qwen-decision-27b
python -m pip install -r requirements.txt
python -u serve_checkpoint.py
```

The single-process server binds `127.0.0.1:8765`. `GET /health` reports the loaded configuration. `POST /v1/systemone` expects model `our-noncausal-27b-step1533` and the normal System One `state` and `questions` fields. Use an SSH tunnel when the evaluator is on another machine. The server is intended for local evaluation, without production authentication or concurrency handling.

`DECISION_ARTIFACT` can select the downloaded artifact directory; default is the server's directory. `DECISION_PROVENANCE` selects the output metadata path; default `/tmp/qwen-decision-provenance.json`. The portability changes affect file locations only. `serve_checkpoint_original.py` is the exact server used in the run; its request-handler AST is identical to the portable version.

For the official benchmark, build the private corpus following the [Decision Index instructions](https://github.com/apolinario/decision-index), use commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`, then run:

```bash
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8765 \
  --option model=our-noncausal-27b-step1533 \
  --option timeout=3600 --suite-dir /path/to/suite-0.3 \
  --out runs/noncausal-step1533 --compact
```

## Frozen inference settings

- Base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Upstream source reference: `perplexity-ai/pplx-decider-v1.1-27b`, revision `3b45dead91dfa6d95aad6b95764a606fab2bf7a6`; locally modified source is included verbatim.
- Noncausal mask in full SDPA attention layers; recurrent linear-attention layers retain their native mask. Last-token pooling, dedicated 255-way readout, no KV cache.
- Maximum 255 options and 8,192 tokens per complete question branch, four questions per forward batch. Inputs are never truncated; excess capacity is rejected explicitly.
- Unchanged `decision_messages` prompt construction across benchmarks. `choice` and `noul` supported by this server; `score` is not exposed here.
- Saved temperature **1.0862525693910237**, fitted on the training calibration split, not the benchmark. No benchmark-specific prompt changes or tuning.

## Training and data disclosure

Step 1,533 selected artifact. LoRA rank 16, alpha 32, dropout .05; 80,997,376 trainable parameters including readout. Microbatch 4 × accumulation 8, learning rate 5e-5, seed 20260922, training sequence limit 512. 48,847 rows available, 49,056 examples seen; 512 development and 512 temperature-fitting rows.

Data preparation used a deterministic two-thirds sample per suite from the archived 0.75-scale text corpus plus the project's synthetic banking gap curriculum. It filtered rendered inputs to 480 tokens to leave a 32-token margin for option shuffling. Namespaced family separation was checked between training/development/calibration splits. This was a fresh attention-ablation experiment, not a byte-identical rerun of historical banking experiments.

**Overlap with Decision Index public datasets has not been fully audited. No contamination-free claim is made.** The general public data mixture may share tasks or examples with public benchmarks; internal split checks do not establish independence from Decision Index. The base model's pretraining overlap is also unknown. The evaluated checkpoint was frozen before this benchmark run. Maintainers should interpret the public result with this disclosure and use their private tests.

`decision_config.json` records data hashes and the exact training configuration. Optimizer state and training data are not included in this inference release. Vision capability is not evaluated or claimed by this text benchmark run.

## Attribution

Upstream AutoJev/decider source and Qwen model are Apache-2.0; see `LICENSE` and `NOTICE`. Project training and evaluation helpers are also distributed with their original MIT notice in `LICENSE.project`. Modified upstream code is included to freeze the inference implementation. This publication is not affiliated with Perplexity or TypeSafe.
