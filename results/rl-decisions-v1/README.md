# Matched decision-policy RL pilot: negative result

The tested RL hybrid did not improve development accuracy or calibrated NLL, Brier, or ECE relative to continued SFT. We retain the starting expanded-data checkpoint; this experiment does not replace the published model or leaderboard submission.

| Model | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Fitted temperature |
|---|---:|---:|---:|---:|---:|
| starting_sft | 86.85% | 0.3419 | 0.1843 | 0.0234 | 1.055 |
| sft | 86.72% | 0.3397 | 0.1830 | 0.0321 | 0.975 |
| rl_hybrid | 86.20% | 0.3515 | 0.1886 | 0.0404 | 1.276 |

RL made four fewer correct decisions than continued SFT and five fewer than the unchanged starting checkpoint, out of 768. This is one small, single-seed pilot, not evidence that RL generally cannot help. There was no public-benchmark evaluation or checkpoint selection in this experiment.

## Matched protocol

Both continuation arms start from the same [expanded-data SFT checkpoint](../targeted-data-v3/README.md) and process the same 1,108 examples, 249,855 nonpadding input tokens and 35 optimizer updates. Model, initialization hash, data hashes, token-plan hash, reference-cache hash, seed, learning rate and batch settings match in the saved configurations. Candidate order is fixed. Each model's temperature is fitted on a separate 768-case calibration fold before scoring the shared 768-case development fold.

The RL loss is negative exact expected correctness reward plus 0.05 KL to the frozen starting policy and 0.25 candidate cross-entropy. Rewards are derived from the same labels as SFT: this changes the objective, without adding teacher information or sampled reasoning. See the [objective, tests and commands](../../experiments/rl_decisions/README.md). This is not a reproduction of an undisclosed RLCD, PPO or GRPO recipe.

The run used one RTX PRO 6000 Blackwell GPU with approximately 96 GB memory; each arm peaked at about 52.3 GiB allocated. Reported timed sections were about 11 minutes per arm, excluding model loading and reference-cache construction. Setup, checkpoint transfer and verification add overhead. Full optimizer/RNG checkpoints and selected artifacts were verified on Drive before releasing the GPU.

## Accuracy by task

| Task | Cases | Starting SFT | Continued SFT | RL hybrid |
|---|---:|---:|---:|---:|
| acos-quad-verification | 55 | 96.36% | 96.36% | 96.36% |
| acos-sentiment | 55 | 96.36% | 96.36% | 96.36% |
| aegis2 | 55 | 87.27% | 87.27% | 87.27% |
| boolq | 55 | 94.55% | 94.55% | 94.55% |
| civil_comments | 55 | 87.27% | 87.27% | 85.45% |
| esci-relevance | 55 | 67.27% | 67.27% | 67.27% |
| massive-de-DE | 55 | 89.09% | 89.09% | 89.09% |
| massive-en-US | 55 | 89.09% | 89.09% | 90.91% |
| multinli | 55 | 94.55% | 94.55% | 92.73% |
| paws | 55 | 92.73% | 94.55% | 94.55% |
| pubmedqa | 55 | 76.36% | 76.36% | 76.36% |
| squad2 | 55 | 87.27% | 87.27% | 87.27% |
| verified-code | 54 | 74.07% | 74.07% | 70.37% |
| vitaminc-dev | 54 | 83.33% | 79.63% | 77.78% |

## Files and interpretation

- `comparison.json`: three-way aggregate and task-level comparison, including soft-label metrics where available.
- `sft-*` and `rl_hybrid-*`: original aggregate reports, configurations and per-update training logs.
- Starting checkpoint report: [`targeted-report.json`](../targeted-data-v3/targeted-report.json).

The higher training reward did not translate to better held-out accuracy. RL required a higher fitted temperature and had worse ECE even after calibration. Soft-label NLL improved slightly, but soft-label Brier did not beat continued SFT; this is not an overall win. Small task slices and one seed do not support strong significance claims. The pilot is too short to establish the best RL objective or hyperparameters.

No raw third-party examples, credentials, private Drive identifiers or model weights are included here. Data sources and reproduction instructions are documented in the preceding data-mixture experiment.
