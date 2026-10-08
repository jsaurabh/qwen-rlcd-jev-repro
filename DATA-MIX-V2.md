# Qwen3.5-9B data-mix v2

## Result

The 70/20/10 curriculum recovered nearly all original Nimble holdout accuracy while retaining most of the synthetic gap performance. It did not improve public JevBench accuracy.

| Model | Nimble holdout | Gap holdout | JevBench | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Original Nimble adapter | 284/324 (87.65%) | — | 184/231 (79.65%) | 0.2886 | 0.1217 |
| Gap curriculum v1 | 277/324 (85.49%) | 427/480 (88.96%) | 185/231 (80.09%) | 0.2604 | 0.0643 |
| Data-mix v2 | 282/324 (87.04%) | 420/480 (87.50%) | 184/231 (79.65%) | 0.2895 | 0.0892 |

JevBench tier results for data-mix v2 were 48/48 easy, 68/72 standard, and 68/111 hard. Relative to the original adapter, intent improved by two and probability by one, while temporal/numeric lost one, adequacy lost one, and adversarial lost one. Relative to gap v1, data-mix v2 gained one each on standard, intent, probability, and multi-hop, but lost two on hard and two on temporal/numeric, plus one each on long-policy and trap.

## Training setup

- Base model: `Qwen/Qwen3.5-9B`, frozen revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- LoRA: rank 16, alpha 32, dropout 0.05, all language-model linear layers
- Data: 2,676 original Nimble rows, 764 synthetic gap rows (382 intact contrastive pairs), and 382 replay rows sampled from original `noul` and `score` examples
- Mix: 70% original, 20% gap, 10% hard replay
- Training: one epoch, 478 optimizer steps, microbatch 2, gradient accumulation 4, 2,048-token cap, bf16
- Runtime: A100 40 GB, 1,607 seconds training, 20.33 GiB peak allocated memory
- Frozen evaluation commits: Nimble `f136b3f75721fda4ea961f73993cc50b08488835`; JevBench `5e95f23cbb7be098a9061fea924c4421620ab1a5`
- Adapter archive SHA-256: `c43828fe7a6005682754878a21b03eebd167cb745e212af81277757a262b903b`

## Interpretation

The mix did what its replay component was intended to do on the in-distribution Nimble holdout: it recovered five of the seven examples lost by gap v1. The public benchmark remained flat because the synthetic curriculum still does not match the hard JevBench temporal/numeric distribution. The next experiment should use error-driven examples modeled on the failure structure without copying benchmark items, preserve the original/replay share, and select checkpoints on a combined validation objective instead of synthetic gap accuracy alone.
