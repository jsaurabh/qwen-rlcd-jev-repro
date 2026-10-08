# Gap curriculum v1

This experiment adds a narrow, rule-grounded curriculum to the frozen public
Nimble training set. It targets the weakest public JevBench families measured in
the first Qwen3.5-9B reproduction: temporal/numeric comparisons, probability
thresholds, and long-policy conjunctions.

## Data and training

- Base: `Qwen/Qwen3.5-9B` at revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Original Nimble training rows: 2,676.
- Added curriculum rows: 1,920, from 960 contrastive pairs.
- Separate curriculum evaluation: 480 rows from 240 pair families.
- Labels: deterministic Python rules; no evaluated-model or Jev labels.
- LoRA: rank 16, alpha 32, dropout 0.05 over all language-model linear layers.
- Optimizer: AdamW, 5e-5 peak learning rate, linear schedule, 58 warmup steps,
  effective batch size 8, 575 optimizer steps, 2,048-token limit.
- GPU: A100 40 GB; peak allocated memory 20.33 GiB.

Colab pruned two ephemeral runtimes. The completed fit resumed from a locally
verified step-400 Trainer checkpoint, preserving model, optimizer, scheduler,
RNG, and Trainer state. Adapter archive SHA-256:
`1613407ae9f91b1b8b7197e75a1b35a3fa44f91a1c7c326e348a3e43d3d30d2c`.

## Results

| Evaluation | Original adapter | Gap adapter | Change |
|---|---:|---:|---:|
| Nimble frozen holdout | 284/324 (87.65%) | 277/324 (85.49%) | -7 |
| Gap holdout | base 246/480 (51.25%) | 427/480 (88.96%) | +181 vs base |
| JevBench public | 184/231 (79.65%) | 185/231 (80.09%) | +1 |
| JevBench Brier (lower is better) | 0.2886 | 0.2604 | -0.0282 |
| JevBench ECE (lower is better) | 0.1217 | 0.0643 | -0.0573 |

JevBench tier accuracy stayed 48/48 on easy and 67/72 on standard. Hard improved
from 69/111 to 70/111. Family changes were mixed:

| Family | Original | Gap adapter | Change |
|---|---:|---:|---:|
| temporal_numeric | 3/15 | 4/15 | +1 |
| long_policy | 10/19 | 11/19 | +1 |
| intent | 22/24 | 23/24 | +1 |
| trap | 7/8 | 8/8 | +1 |
| adequacy | 9/12 | 8/12 | -1 |
| adversarial | 4/6 | 3/6 | -1 |
| multi_hop | 15/18 | 14/18 | -1 |

## Decision

Keep this adapter as an experiment, not an unconditional replacement. It raises
public JevBench accuracy by one task and materially improves calibration, but it
loses seven tasks on Nimble's frozen holdout. The narrow curriculum teaches its
own held-out templates well, yet transfer to the intended weak families is only
modest. The next dataset revision should broaden surface forms and causal
structures, reduce repeated template signatures, and include explicit replay or
sampling weights that protect the original score/noul capabilities.

Artifacts:

- Adapter: `colab-results/nimble-gap-9b-v1-adapter/`
- Archive and JevBench output: `colab-results/nimble-gap-9b-v1/`
- Dataset: `data/jev_gap_curriculum_v1/`
- Generator: `tools/generate_gap_curriculum.py`
