# Perplexity decider v1.1: exact reference and feasible reproduction

Perplexity released a strong open reference implementation at
[`perplexity-ai/pplx-decider-v1.1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b).
It reports a 61.56 Decision Index, compared with 57.9 for Jev in that suite.
The repository includes the evaluated inference and training source, pinned
configuration, data-source manifest, checkpoint checksums, and evaluation
report.

## What the released run actually used

| Parameter | Released value |
|---|---:|
| Base model | `Qwen/Qwen3.8-27B` at revision `1d4bf0f...` |
| Update | Full-parameter supervised fine-tuning, not LoRA |
| Attention | Noncausal in full-attention layers; native recurrence in linear-attention layers |
| Readout | Separate BF16 `255 x 5120` linear decision head |
| Pooling | Last position |
| Training rows | 626,033 |
| Tasksource rows | 530,103 (84.68%) |
| Effective batch | 256 |
| Optimizer steps | 2,446, one epoch |
| Context limit | 8,192 tokens |
| Learning rate | `2e-6`, 15% warmup, 1% weight decay |
| Optimizer | FP32 AdamW masters and state offloaded to CPU |
| Training machine | 8 GPUs, 1.5 TB host RAM, 96 CPUs |
| Calibration | Scalar temperature `1.0087417621` |

The release does **not** contain the 626,033-row JSONL training corpus. It
publishes hashes, row counts, and source counts. In particular, the Tasksource
portion passed a GLM filtering step whose selected `kept.jsonl` is not part of
the model repository. Exact from-base retraining therefore requires either that
frozen corpus or a byte-identical reconstruction of the filtering step.

## Exact inference check on the frozen banking sample

We downloaded the native 52.2 GB release, imported the inference code included
inside the checkpoint, preserved its noncausal attention hook and saved
temperature, and evaluated the identical 128 cases used in our earlier
three-way comparison.

| Model | Accuracy | Brier | NLL | ECE |
|---|---:|---:|---:|---:|
| Jev 1.13.0 | **98.44%** | **0.0395** | **0.0901** | 0.0624 |
| Perplexity decider v1.1 27B | 92.19% | 0.1066 | 0.1588 | 0.0532 |
| Our banking LoRA 27B | 91.41% | 0.1072 | 0.1527 | **0.0205** |
| Public AutoJev 27B | 89.06% | 0.1421 | 0.2351 | 0.0824 |

This is a narrow synthetic banking test, not the Decision Index. The complete
per-case output is committed at
[`results/colab/pplx-decider-v1.1/banking-128.json`](../results/colab/pplx-decider-v1.1/banking-128.json).

Run the same checkpoint evaluation with:

```bash
python -u colab/eval_pplx_decider_v11.py \
  --eval-file data/jev_gap_curriculum_v1/eval.jsonl \
  --prior-report results/colab/autojev-27b-v4/three-way-comparison.json \
  --output runs/pplx-decider-v1.1-banking-128.json
```

Expect roughly 49 GiB for weights and at least several more GiB of working
memory. Our G4 run used 49.26 GiB of allocated GPU memory and evaluated 128
rows in 35.3 seconds after model loading.

## Single-GPU training ablation

Our trainer now accepts Perplexity's core attention change:

```bash
python -u colab/autojev_lora_train.py \
  --train-file /path/to/train.jsonl \
  --dev-file /path/to/dev.jsonl \
  --temperature-file /path/to/temperature.jsonl \
  --attention-mode noncausal_full_attention \
  --max-length 512 \
  --micro-batch 4 \
  --grad-accum 8 \
  --steps 1533 \
  --out /path/to/output
```

This preserves the published prompt, 255-way readout, last-position pooling,
noncausal SDPA behavior, recurrence in linear-attention layers, supervised
candidate loss, and held-out temperature fit. It remains a LoRA approximation:
80,997,376 trainable parameters, 512-token branches, effective batch 32, and
our smaller public data mix.

For a clean causal-mask ablation, hold every argument, data file, seed, and row
order fixed and run once with `--attention-mode causal` and once with
`--attention-mode noncausal_full_attention`. Compare the same held-out cases,
including option-order perturbations, before adding new Tasksource data.

## What it would take to match 61.56 from base

1. Reconstruct and freeze the 626,033-row corpus, including the unpublished GLM
   filtering decisions, and match every published SHA-256 hash.
2. Provision one eight-GPU node with enough aggregate GPU memory for BF16 model,
   gradients, and activations plus about 1.5 TB of CPU RAM for FP32 masters and
   AdamW moments.
3. Use the released source and exact `training/config.json` settings for 2,446
   synchronized updates.
4. Verify the selected checkpoint on the frozen 1,497-row development fold,
   where the release reports 90.45% accuracy, 0.1395 Brier, 0.2658 NLL, and
   0.0159 ECE.
5. Run the complete Decision Index 0.2.1 suite. Our 128-case banking comparison
   cannot establish a match to the published 61.56 score.
