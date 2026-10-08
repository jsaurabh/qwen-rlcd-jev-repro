# Banking demo with Qwen2.5-1.5B

Installed and verified on the 16 GiB M1 Pro. The model is the 4-bit MLX conversion of Qwen2.5-1.5B-Instruct, revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.

Use the base model:

```sh
cd qwen-rlcd-jev-repro
.venv/bin/python mlx_jev_repro.py --model models/qwen-1.5b-4bit --max-length 384 \
  --state "A customer reports two unauthorized transfers and another is pending."
```

Compare the MLX runner with official Jev on exactly the same state:

```sh
.venv/bin/python mlx_jev_repro.py \
  --model models/qwen-1.5b-4bit \
  --adapter runs/banking-1.5b \
  --max-length 384 \
  --compare-jev \
  --state "A customer reports an unauthorized wire is pending."
```

The result contains `local_qwen` and `jev`, each with the same `scores` structure for department, urgency, and escalation. Every score includes the selected/top choice, directly comparable `top_probability`, and complete candidate distribution. Jev's native Choice/Score concentration measure is retained separately as `provider_confidence`; it is `null` for local Qwen and for Noul, which has no separate confidence. Urgency's `value` is the probability-weighted score; escalation's `value` is the probability of true. Backend model metadata and elapsed time are included, and Jev additionally reports API usage. The key is read from `TYPESAFE_API_KEY` or the ignored project `.env` and is never emitted.

Without `--compare-jev`, `mlx_jev_repro.py` remains local-only. Override the Jev alias with `--jev-model` if needed.

Use the experimental 1.5B adapter by adding `--adapter runs/banking-1.5b`. Do not load the old 0.5B adapter into 1.5B. The base-model default in the script remains 0.5B; always specify `--model` for 1.5B.

Repeat training:

```sh
.venv/bin/python mlx_jev_repro.py --model models/qwen-1.5b-4bit \
  --max-length 384 --out runs/banking-1.5b-repeat
```

Download again if needed (existing virtual environment):

```sh
.venv/bin/hf download mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --revision 8b403126fc14f14cfc99bb4cfa72ecbc129ea677 --local-dir models/qwen-1.5b-4bit
```

The banking schema is the current `mlx_jev_repro.py` default. Its synthetic generator uses banking cases with corresponding department, urgency, and review labels. The earlier pre-banking-schema 0.5B report is preserved at `results/mlx/smoke-0.5b-legacy-report.json`.

The run used 12 supervised plus 4 bandit updates, accumulation 4, rank-4 LoRA on the last four blocks, and 77,824 trainable parameters. Prompts reached 190 tokens (384-token ceiling). Training took 21.32 seconds; total after imports was 35.79 seconds. Warm three-field inference median: 281.9 ms. MLX tracked 1.271 GB peak allocation, excluding Python and allocator caches. This run used longer banking prompts, so timings cannot isolate the effect of model size against the prior 0.5B run.

Finite gradients, adapter updates and reload, typed outputs, normalization and cache checks passed. A fresh process successfully loaded the 1.5B adapter. Results and data are under `runs/banking-1.5b/`; provenance records the exact model revision and script hash.

On only 24 synthetic test decisions, sampled-label accuracy changed from 0.667 to 0.583 after training. Supervised training improved Brier loss (0.522 to 0.487), but the subsequent four RL updates worsened it to 0.563. Temperature scaling chose T=2 on validation; post-temperature Brier was 0.527. This establishes compatibility, not banking reliability or improved accuracy. All data is synthetic and routing policies are illustrative.
