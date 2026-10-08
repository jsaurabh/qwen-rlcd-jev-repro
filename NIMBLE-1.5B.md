# Nimble recipe scaled to Qwen2.5-1.5B on an M1 Pro

This track ports Bespoke Nimble's public candidate-classification recipe to the existing local 4-bit Qwen2.5-1.5B model. It preserves the frozen 2,676-example contrastive training set, untouched 324-example holdout, evidence certificates, source-family separation, one-token A–Z candidates, candidate-only cross-entropy, seeded candidate permutation, effective batch size 8, learning rate `5e-5`, 10% warmup, linear decay, one epoch, prompt contract, and before/after evaluation.

The port is reproducible, runs successfully on the 16 GiB M1 Pro, and should **not** be deployed. Both completed adapters failed the frozen holdout and the stronger adapter collapsed to always selecting candidate position B.

## Integrity and contract

The local Nimble checkout is pinned at commit `f136b3f75721fda4ea961f73993cc50b08488835`. Its offline verifier passed:

- 2,676 training records, or 1,338 base/counterfactual pairs
- 324 holdout records, or 162 pairs
- every evidence certificate passed
- every pair changes its target
- no source-family overlap
- frozen SHA-256 values match Nimble's published manifest

Qwen2.5 tokenization successfully encodes every retained prompt with distinct, ordinary one-token candidate codes. The longest training prompt is 1,094 tokens, so the local 1,152-token contract includes every record without truncation.

The intentional hardware adaptations are a 4-bit Qwen2.5-1.5B base and MLX training. The stronger run otherwise uses Nimble's rank 16 across every attention and MLP projection in all 28 blocks. It exposes 18,464,768 trainable parameters and produces a 70 MB adapter.

## Commands

```sh
cd qwen-rlcd-jev-repro

# Verify Nimble's retained dataset.
PYTHONPATH=reference-nimble .venv/bin/python -m nimble.training.verify_dataset

# Small-capacity diagnostic run: rank 8 over the last 8 blocks.
.venv/bin/python nimble_1_5b_mlx.py \
  --out runs/nimble-1.5b-v1

# Faithful-capacity run: rank 16 over all 28 blocks.
.venv/bin/python nimble_1_5b_mlx.py \
  --rank 16 --layers 28 \
  --out runs/nimble-1.5b-r16-all-v1
```

The small run took 2,138 seconds of training compute. The rank-16 run took 3,768 seconds, or about 62.8 minutes. Both use microbatch one with eight-step accumulation and see all 2,676 examples exactly once.

## Frozen synthetic holdout

| Model | Accuracy | NLL | Brier | Pair accuracy | Decision behavior |
|---|---:|---:|---:|---:|---|
| Base Qwen2.5-1.5B | 44.14% | 2.749 | 0.980 | 1.23% | varied candidates |
| Rank 8, last 8 blocks | 41.67% | 1.074 | 0.636 | 0.62% | strong B/C bias |
| Rank 16, all blocks | 33.33% | 1.100 | 0.651 | 0.00% | selected B on 324/324 |

The lower NLL does not indicate a better decision model. Training reduced extreme wrong probabilities while converging toward a position prior. The shuffled-order test confirms the collapse: the rank-16 adapter still selects position B on every record and reaches 33.02% accuracy.

## External human-labeled test

The public VitaminC development benchmark was downloaded from its canonical source and converted with Nimble's committed converter. It contains 599 records from 171 contrastive families and was never used for training or model selection.

| Model | Accuracy | NLL | Brier | Whole-family accuracy | Selected positions |
|---|---:|---:|---:|---:|---|
| Base Qwen2.5-1.5B | 55.09% | 2.241 | 0.808 | 2.92% | A: 551, B: 48 |
| Rank-16 adapter | 38.73% | 1.146 | 0.695 | 0.00% | B: 599 |

The adapter is rejected. The external result shows the same position collapse as the synthetic holdout.

## What transfers and what does not

Nimble's engineering and evaluation controls transfer well: contrastive minimal pairs, evidence-deletion certificates, family-level split isolation, prompt/model contracts, candidate permutation, untouched final evaluation, and external human-labeled benchmarks. These controls found a failure that aggregate loss alone would have hidden.

The broad ten-domain one-epoch curriculum does not transfer to this 1.5B model. The published Nimble checkpoint uses Qwen3.5-9B, whose base holdout agreement is much higher. On the smaller base, minimizing one epoch of candidate cross-entropy finds a low-capacity shortcut instead of learning the paired evidence rules. Raising LoRA capacity does not fix the base-model capability gap.

The next useful 1.5B experiment is a banking-only contrastive curriculum with whole policy/product families held out, pair-aware validation, and a deployment gate requiring improvement in both individual accuracy and whole-pair accuracy under at least two candidate permutations. The existing banking v2 adapter remains the best local checkpoint; its synthetic performance still requires independently authored or real de-identified banking evaluation.
