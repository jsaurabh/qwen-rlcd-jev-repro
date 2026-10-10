# Full-epoch RL versus continued SFT

Both continuations improve the starting checkpoint. RL does not establish a clear advantage over matched continued SFT: it gets two more of 768 decisions correct, with slightly better NLL and ECE but slightly worse Brier. This is a single-seed development-set ablation, not a new public leaderboard score.

| Model | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Starting expanded-data SFT | 86.85% | 0.34188 | 0.18426 | 0.02338 |
| Continued SFT | 88.15% | 0.32460 | 0.16860 | 0.02914 |
| RL hybrid | 88.41% | 0.31965 | 0.16933 | 0.02356 |

RL gets 679/768 correct; continued SFT gets 677/768; the starting checkpoint gets 667/768. Temperature is fitted independently on the calibration fold: 1.44697 for RL, 0.99329 for SFT. On the 110 cases with soft targets, RL has soft NLL 0.33729 versus SFT 0.34356, while SFT has lower soft Brier (0.05134 versus 0.05554).

## Matched setup

Each arm processes the same 92,809 examples once: 20,601,781 nonpadding tokens, 2,901 updates, seed 20261010, context 512, microbatch 4, accumulation 8, learning rate 1e-5 with 5% token warmup and linear decay. Both start from the same expanded-data SFT adapter/readout and reset the optimizer at the start of the new experiment. The base is Qwen/Qwen3.8-27B, with rank-16 LoRA and a 255-way readout; only full-attention SDPA layers are made noncausal. The base weights stay frozen. Candidate order is fixed and identical across arms. Initialization, data, reference-cache and example-plan hashes are recorded in the run configurations.

RL uses exact expected correctness reward plus 0.05 KL to the frozen starting policy and 0.25 candidate cross-entropy. Continued SFT uses candidate cross-entropy. Both see the same labels: RL introduces no additional teacher information. This is one-step finite-action policy optimization, not PPO/GRPO or a claimed replication of undisclosed Matilda RL/RLCD. [Code and reproduction commands](../../experiments/rl_decisions/README.md#full-epoch-comparison).

## Learning curves

Diagnostics use the same 768 development cases repeatedly, with temperature fitted on a separate 768 cases. They do not choose a checkpoint or stop training. The final artifact is always update 2,901.

| Update | SFT accuracy | RL accuracy | SFT NLL | RL NLL |
|---|---:|---:|---:|---:|
| 500 | 86.46% | 86.98% | 0.33285 | 0.33338 |
| 1000 | 87.24% | 87.50% | 0.31963 | 0.32073 |
| 1500 | 88.02% | 88.02% | 0.31818 | 0.32518 |
| 2000 | 88.54% | 88.15% | 0.32099 | 0.31873 |
| 2500 | 88.28% | 88.15% | 0.32562 | 0.31964 |
| 2901 | 88.15% | 88.41% | 0.32460 | 0.31965 |

The earlier 35-update pilot was negative; increasing the budget improved both arms. That does not isolate RL as the source of the gain. [Preserved pilot](../rl-decisions-v1/README.md).

## Task breakdown

| Suite | Cases | Starting accuracy | SFT accuracy | RL accuracy |
|---|---:|---:|---:|---:|
| acos-quad-verification | 55 | 96.36% | 96.36% | 98.18% |
| acos-sentiment | 55 | 96.36% | 96.36% | 96.36% |
| aegis2 | 55 | 87.27% | 85.45% | 85.45% |
| boolq | 55 | 94.55% | 94.55% | 94.55% |
| civil_comments | 55 | 87.27% | 87.27% | 87.27% |
| esci-relevance | 55 | 67.27% | 69.09% | 69.09% |
| massive-de-DE | 55 | 89.09% | 92.73% | 92.73% |
| massive-en-US | 55 | 89.09% | 90.91% | 92.73% |
| multinli | 55 | 94.55% | 96.36% | 94.55% |
| paws | 55 | 92.73% | 96.36% | 98.18% |
| pubmedqa | 55 | 76.36% | 76.36% | 78.18% |
| squad2 | 55 | 87.27% | 90.91% | 90.91% |
| verified-code | 54 | 74.07% | 74.07% | 74.07% |
| vitaminc-dev | 54 | 83.33% | 87.04% | 85.19% |

Versus SFT, RL gains one correct answer each on ACOS quad verification, English MASSIVE, PAWS and PubMedQA, and loses one each on MultiNLI and VitaminC. Other suite accuracy totals tie. Each suite has only 54–55 cases. The aggregate NLL advantage is concentrated in verified-code (0.75851 RL versus 0.94240 SFT), despite identical accuracy there; it is not a uniform task-level gain. Full raw and calibrated suite metrics are in the report JSON files.

## Runtime, recovery, and persistence

One RTX PRO 6000 Blackwell Server Edition (96 GB) ran the arms sequentially. Summing all 2,901 unique training-log entries gives 6.8836 hours for RL and 6.8828 hours for SFT, excluding reference generation, evaluation, checkpoint transfers and provisioning. Typical throughput was about 820–850 nonpadding tokens/s; peak PyTorch allocated memory was 52.29 GiB (RL) and 52.30 GiB (SFT). This is a training measurement, not inference latency.

SFT reports 28,408.35 seconds (7.89 hours) from training start through final evaluation; final selected-artifact upload is outside that timer. RL report `wall_seconds=5151.50` covers only its last resumed segment (2,400–2,901 plus final evaluation), and must not be presented as total RL runtime. Its original runtime disappeared after the latest recoverable checkpoint at 1,900; the cause was not confirmed. Recovery restored the optimizer, RNG, frozen reference and data plan. A later checkpoint-upload timeout at 2,400 was recovered on the same VM, with bounded upload retries. The retained training logs contain all steps exactly once; computation after the last durable checkpoint on the lost VM, if any, cannot be accounted for.

Both selected artifacts and metadata were checksum-verified on Drive. The runtime credential was removed and the GPU released. No model weights, account credentials, Drive receipts or raw corpus are published here. Starting artifacts and rebuilt data must be staged separately; aggregate data hashes allow checking whether a rerun matches.

## Limits

One seed, 768 development cases, repeated diagnostics, and no paired per-example significance test: the two-answer advantage is insufficient to claim a reliable improvement. Calibration metrics are distribution-dependent. These results do not change the published Hugging Face model or Decision Index submission. The experiment tests our explicit objective and budget, not every possible RL method.
