# Online decision-policy RL on ALFWorld

This experiment changes the source of the learning signal: the policy interacts with a stateful executable environment and receives terminal task success, not a reward vector computed from classification labels. It is a finite-action policy, preserving the decision model's candidate readout. It does not generate chain-of-thought. It is not a reproduction of an undisclosed Matilda, Jev or RLCD recipe.

## Scientific question

Can outcome-driven, multi-step reinforcement learning improve the 27B decision policy's held-out task completion while preserving its classification quality? Improvement on ALFWorld alone is not evidence of improvement on Decision Index. A later independent transfer evaluation is necessary before making that claim.

## Environment and information boundary

Use ALFWorld's text simulator, pinned at aaba6870f86c5be6a08a491f32a50b906227bc3e, with TextWorld 1.7.0 in a separate Python 3.11 environment. Train only on official train games. Freeze balanced 24-case samples of valid_seen and valid_unseen before optimization. Use public solvable-game metadata solely for catalog eligibility, consistent with ALFWorld's loader. The agent receives only initial observations, a bounded recent interaction history and the simulator's admissible actions. No expert action, walkthrough, hidden facts, planner, or gold trajectory is passed to the agent or loss. This admissible-action setting is easier than unconstrained command generation and must be reported as such.

The original task description and all current candidates are preserved. If the 1,536-token context budget is exceeded, remove oldest complete history entries; never silently truncate the task or current choices. Fail if even the irreducible prompt does not fit. The episode horizon is 50 actions. Candidate order is randomized independently; baseline and final evaluation use the same candidate-order random seed and greedy action selection.

## Algorithm

REINFORCE leave-one-out with two tasks per update and four independent sampled trajectories per task. One optimizer step per fresh batch of trajectories; no replay, PPO epochs or supervised answer loss.

For trajectory i:

    return_i = task_success - 0.002 * number_of_actions
               - 0.005 * sum_t log(policy(action_t|state_t) / reference(action_t|state_t))
    advantage_i = return_i - mean(return_j for the OTHER three trajectories on the same task)
    loss = -mean_over_trajectories(advantage_i * sum_t log policy(action_t|state_t))

The sampled KL term regularizes the trajectory policy against its frozen initialization. Rewards and leave-one-out baselines are detached. Credit is propagated to every sampled action in the trajectory. The baseline excludes the trajectory itself. Do not divide each trajectory by its length or normalize advantages by within-group standard deviation: both change the stated estimator. Terminal success remains the primary reported metric, separately from the cost/KL-shaped return.

A tiny two-step MDP test exhaustively averages independent trajectory groups and checks the resulting gradient against the analytical expected-success gradient. GPU preflight checks reference-policy equality, frozen reference gradients, finite actor gradients, and recomputation agreement before evaluation. During updates, reject rollout/recomputed selected log-probability differences above 0.15 rather than silently applying off-policy gradients.

## Model and budget

Initialize from the completed full-epoch SFT adapter/readout (88.15% on our development set), not the prior RL-hybrid checkpoint. Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; rank-16 LoRA plus 255-way readout; frozen base; noncausal full SDPA attention with native recurrent layers unchanged. Share one base between trainable and frozen-reference adapters, with a separate frozen readout. Disable dropout for consistent rollout and update probabilities. AdamW learning rate 2e-6, weight decay 0, gradient clipping 1.0; microbatch 4; seed 20261010.

Budget: 64 updates (up to 512 training episodes), or six hours in the training loop, whichever comes first; baseline/final evaluation and setup are additional. After update 8, stop if fewer than four successful trajectories or fewer than two groups with mixed success have occurred. This is a training-signal gate, not selection on held-out outcomes. Report any early stop explicitly. No guaranteed improvement is assumed.

## Evaluation and persistence

Run baseline/final greedy task success on exactly the same fixed official validation tasks. Report paired success counts, mean actions and per-task-type breakdowns. These small samples are a feasibility pilot, not a full ALFWorld leaderboard evaluation. Evaluate the same 768 development decisions and separate 768 calibration examples before/after, reporting accuracy, NLL, Brier, ECE and fitted temperature. Temperature fitting is only for the classification diagnostic; it does not change the RL sampling policy.

A separate privileged controller holds the explicitly authorized Drive credential. The training worker and simulator run as UID65534 with a private HOME and no credential access. Complete checkpoints include adapter/readout, optimizer, RNG, configs and trajectory logs. Upload and checksum-verify checkpoint zero, every four updates and final selected artifact before proceeding. Retain only two verified periodic checkpoints locally. Store dependencies and code snapshot on Drive. Delete the runtime credential on controller exit and release GPU after final backup verification. Do not download weights to the laptop.

References:
- https://github.com/alfworld/alfworld
- https://aclanthology.org/2024.acl-long.662/
- https://huggingface.co/docs/trl/rloo_trainer

## Operational files

`bootstrap_colab.py` provisions Python dependencies and text-game assets; `requirements-colab.txt` pins the model environment. The first live run needed `relocate_env_python.py` because uv initially installed its interpreter under `/root`, inaccessible to UID65534. The saved bootstrap now specifies a public runtime install directory to avoid that problem on fresh runs. This changes only interpreter location, not training behavior.

`controller.py` is the run controller, parameterized through environment variables; no account-specific Drive identifiers are distributed. `train_online.py`, `env_worker.py` and `policy_objective.py` contain the experiment. `summarize_results.py RESULTS_DIRECTORY` calculates paired success gains/losses and an exact descriptive McNemar test from completed baseline/final trajectory logs. It also includes retention metrics and the actual stopping condition. Seven objective, analytical-gradient and variable-length batching tests passed in the GPU environment. The paired report also has a local smoke check.

## First live recovery

The first rollout completed seven of eight tasks, but training stopped before any optimizer update when trajectory-major recomputation changed microbatch padding and a selected action log-probability differed by 0.18875. The guard threshold stayed at 0.15. The correction reconstructs the exact time-major active-row microbatches used by rollout, including batch membership and padding shape. A regression test covers variable episode lengths. Resume from checkpoint zero restores optimizer and RNG state, validates the initial reference, retention data and checkpoint against Drive manifests, and reuses the frozen baseline. This is an implementation correction, not a change to rewards or hyperparameters. Original failure logs are preserved.

Recovery validation: the first optimizer update completed on 123 transitions, with maximum selected log-probability mismatch 2.384e-07, seven of eight training tasks completed, 235.47 seconds including rollout and update, and 56.38 GiB peak allocation. This verifies the batching fix; training success is not evidence of held-out improvement.

## Reproduce on Colab

The full pilot needs a 96 GB GPU: measured peak allocated memory was about 57.2 GiB, with about 64.2 GiB reserved. The 27B BF16 setup does not fit unchanged on a T4 or 40 GB A100. CPU-only ALFWorld runs in its own Python 3.11 environment. The GPU environment uses the versions in `requirements-colab.txt`. The measured run used the correct but slower PyTorch causal-convolution fallback; it did not require building `causal-conv1d`.

Prerequisites are the SFT `selected` artifact from the [full-epoch experiment](../rl_decisions/README.md#full-epoch-comparison) and its matching `ready` development/calibration data, each stored with a `DRIVE_COMPLETE.json` checksum manifest using `drive_checkpoint_store.py`. These prior artifacts are inputs, not bundled weights or a new claim that initialization is identical to public Perplexity weights. Start from your reproduced SFT checkpoint; different initialization means a different experiment. Do not put a service credential in the repository.

Clone this repository into `/content/qwen-rlcd-jev-repro`, then stage the files in a Colab terminal:

```bash
REPO=/content/qwen-rlcd-jev-repro
EXP="$REPO/experiments/online_rl_alfworld"
RUN=/content/online-rl-v3
mkdir -p "$RUN/code/autojev"
cp "$REPO"/colab/noncausal_27b/autojev/*.py "$RUN/code/autojev/"
cp "$REPO/colab/noncausal_27b/autojev_lora_train.py" "$RUN/code/"
cp "$EXP"/{train_online.py,env_worker.py,policy_objective.py,checkpoint_mailbox.py,test_policy.py,test_policy_torch.py} "$RUN/code/"
cp "$EXP"/{controller.py,drive_checkpoint_store.py} "$RUN/"
cp "$EXP/requirements-colab.txt" /content/online-rl-requirements.txt
python "$EXP/bootstrap_colab.py"
tail -f "$RUN/setup.log"
```

The bootstrap launches setup in the background. Proceed only after `setup-status.json` says `ready_for_restore`. It installs the simulator dependencies and downloads only the official text-game assets, not visual detectors. The bootstrap now explicitly pins TextWorld 1.7.0 and locates the managed Python interpreter outside `/root`; those are reproducibility fixes matching the executed environment.

Provide an authorized Drive credential at `/content/.checkpoint-auth/credentials.json`, owned by root, with directory mode `0700` and file mode `0600`. The controller holds it; the trainer and simulator run as UID 65534 and cannot read it. Set your own folder IDs, then launch:

```bash
export DRIVE_ROOT_FOLDER=YOUR_EXPERIMENT_PARENT_FOLDER
export INITIAL_CHECKPOINT_FOLDER=YOUR_COMPLETED_SFT_SELECTED_FOLDER
export RETENTION_DATA_FOLDER=YOUR_MATCHING_READY_DATA_FOLDER
export EXPERIMENT_NAME=online-rl-alfworld-v3
nohup python -u /content/online-rl-v3/controller.py \
  > /content/online-rl-v3/controller.log 2>&1 &
tail -f /content/online-rl-v3/train.log
```

Use a fresh output directory/runtime for a new run: existing checkpoint receipts must never be reused for different weights. The shipped controller only supports the checkpoint-zero recovery used in this pilot. Although every later checkpoint includes optimizer/RNG state, later-step recovery additionally needs validated cumulative counters, copied logs and the original elapsed-time budget; do not assume it is automatic. The controller deletes the credential on exit. After status becomes `complete`, verify both `selected` and `final-metadata` manifests, then release the GPU; the controller does not stop the Colab assignment itself.

To summarize downloaded compact metadata:

```bash
python summarize_results.py /path/to/final-metadata
# Or recompute the committed summary from compact public outcomes:
python summarize_results.py ../../results/online-rl-alfworld-v3
```

The public results include aggregate metrics, paired task outcomes, training diagnostics and a sanitized split manifest. Full observation/action trajectories and model weights stay in the experiment artifact store. No Decision Index evaluation or leaderboard submission is implied by this ALFWorld result.
