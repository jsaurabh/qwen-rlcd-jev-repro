# Outcome-based online RL: ALFWorld feasibility pilot

The 27B decision model completed 64 online REINFORCE leave-one-out updates using real ALFWorld transitions and terminal task success. It gained four successful tasks on the 24-case seen-environment evaluation and one net success on the 24-case unseen-environment evaluation. Classification accuracy fell by two net correct answers out of 768. This supports further investigation of environment-based RL; it does not establish a Decision Index gain or superiority to Jev.

| Evaluation | Before RL | After RL | Difference |
|---|---:|---:|---:|
| ALFWorld valid_seen, 24 tasks | 17/24 (70.83%) | 21/24 (87.50%) | +4 tasks |
| ALFWorld valid_unseen, 24 tasks | 14/24 (58.33%) | 15/24 (62.50%) | +1 task |
| Classification accuracy, 768 decisions | 88.151% | 87.891% | −0.260 percentage points |
| Classification NLL | 0.324600 | 0.324268 | −0.000332 |
| Classification Brier | 0.168602 | 0.169543 | +0.000941 |
| Classification ECE | 0.029136 | 0.023612 | −0.005524 |

Seen tasks: four gains, no losses; exact paired McNemar p=0.125. Unseen tasks: two gains and one loss; p=1.0. These descriptive tests emphasize the uncertainty of a 24-task sample, with one seed and no independent replication. Confidence metrics have mixed changes; lower ECE alone does not establish better calibration. Each classification arm has temperature fitted on the same separate 768-example calibration fold (0.993290 before, 1.012432 after).

## What ran

- Initial policy: the completed full-epoch **SFT** adapter/readout, not the earlier label-reward hybrid or the released Perplexity checkpoint.
- Frozen Qwen/Qwen3.8-27B base; rank-16 LoRA and 255-way readout, 80,997,376 trainable parameters. Full SDPA layers are noncausal; native recurrent layers retain their original behavior.
- 64 updates, two sampled official training tasks per update, four independently sampled episodes per task: **512 training episodes, 12,513 action transitions**. Task draws can repeat.
- Reward: terminal success minus 0.002/action and a 0.005 sampled trajectory log-policy/reference penalty. No classification-label reward, CE/SFT term, oracle plan, replay or chain-of-thought generation.
- LR 2e-6; one optimizer step per rollout batch; gradient clipping 1; seed 20261010; horizon 50; context 1,536; microbatch 4.
- One RTX PRO 6000 Blackwell 96 GB. Training loop including periodic verified backups took **5.53 hours**; summed rollout/update compute was **5.23 hours**. Baselines, final evaluation, setup and the pre-update recovery are additional. Peak allocated GPU memory: **57.19 GiB**.
- Checkpoints at zero/every four updates/final were verified on Drive, including optimizer and RNG. Final `selected` contains 21 verified files; final metadata contains 12. Runtime credential was deleted and GPU released after verification.

## Evaluation boundary

Both before/after evaluations use the identical frozen task IDs, greedy actions and the same candidate-order RNG seed. The policy sees admissible commands, the initial goal/observation and recent interactions. This is easier than free-form command generation. Candidate shuffling consumes a trajectory-dependent random stream: equal seed does not imply identical menus after policy trajectories diverge.

The evaluation is balanced across six task types (four tasks each), and training never uses validation tasks. `split_manifest.json` contains the official relative game paths and SHA-256 hashes, with local paths removed. The checkpoint at the fixed update budget was evaluated; no best-checkpoint selection on these held-out tasks was performed. The existing classification development fold is a retention diagnostic, not a new blind benchmark.

The strongest next experiment is a larger, independently seeded task-success evaluation, followed by RL environments aligned with retrieval/tool decisions and a frozen Decision Index language/retrieval/tools transfer check. ALFWorld gains alone cannot answer whether this model beats Jev.

## Files and reproduction

- [Training protocol and exact setup](../../experiments/online_rl_alfworld/README.md).
- `experiment.json`: sanitized executed configuration.
- `baseline.json`, `final.json`: per-task-type and classification metrics.
- `paired-outcomes.jsonl`: the 48 paired outcomes and step counts, without observation/action text.
- `training.jsonl`: all 64 updates; training success varies with sampled tasks and is not held-out performance.
- `summary.json`: paired counts, descriptive significance tests and retention metrics.
- `split_manifest.json`: training/evaluation game identifiers and hashes, not the corpus itself.

The first rollout failed a recomputation-agreement guard before any optimizer update. Reconstructing exactly the rollout's time-major microbatch membership fixed BF16/padding differences; all subsequent selected-action log-probability differences were below 5e-7. The guard and RL objective were unchanged. Checkpoint-zero recovery reused the frozen baseline and restored optimizer/RNG state. See the recipe for exact details and later-step recovery limitations.

Full trajectories, model weights, optimizer state and account-specific artifact identifiers are intentionally excluded from this public package. The same initialization must be independently reproduced from the earlier SFT experiment for a matched replication.
