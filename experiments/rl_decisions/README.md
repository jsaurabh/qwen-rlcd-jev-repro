# One-step decision-policy RL experiments

This is a bounded comparison of continued SFT against reward-based policy optimization, starting from the selected expanded-data checkpoint in [the data-mixture pilot](../../results/targeted-data-v3/README.md). The completed pilot found no improvement from this RL hybrid: 86.20% accuracy versus 86.72% for continued SFT and 86.85% for the starting checkpoint. [Full results and limitations](../../results/rl-decisions-v1/README.md).

For a state/schema input, the model's actions are the finite candidate answers. A correct action earns reward 1 and an incorrect action 0. With a soft target distribution, each action's reward is its probability of correctness under that distribution. All action rewards are known from the training labels, so we enumerate expected reward instead of sampling rollouts.

The hybrid objective is:

```text
loss = -sum_a p(a) * reward(a)
       + 0.05 * KL(p || starting_policy)
       + 0.25 * cross_entropy(target, p)
```

This is one-step contextual-bandit policy optimization with an exact expected-reward gradient. It is not GRPO, PPO, or a claimed reproduction of RLCD. Because rewards come from the same labels as SFT, it introduces no additional information: the experiment tests whether the different objective helps. The supervised term and reference-policy penalty constrain drift; neither guarantees calibration.

The comparator continues ordinary candidate cross-entropy training from exactly the same saved adapter/readout. Both arms use the same example order, fixed candidate order, approximately 250,000 nonpadding tokens, seed 20261010, learning rate 1e-5, microbatch 4, accumulation 8 and context 512. Optimizers reset for this new continuation experiment. Initialization is separate from exact optimizer resume. Results must be compared with both continued SFT and the unchanged starting checkpoint.

## Reproduce the 35-step pilot

Install the [27B runtime dependencies](../../colab/noncausal_27b/requirements.txt) and run `test_objective.py` and `test_reference_cache.py`. These cover policy gradients, CE/KL finite differences, reference detachment, padding, partial-batch weighting, cache integrity and resume requirements.

Stage the selected checkpoint and tokenized `ready` data produced by the prior data-mixture experiment. With Drive mounted, run from this directory:

```bash
export PYTHONPATH=../../colab/noncausal_27b:../targeted_data_v3
python train_rl.py \
  --init-artifact /content/drive/MyDrive/decision-data-pilot/targeted/selected \
  --data /content/data-v3/work/ready --train-file targeted-train.jsonl \
  --reference-cache /content/drive/MyDrive/rl-pilot/reference.pt \
  --objective sft --token-budget 250000 --seed 20261010 \
  --micro-batch 4 --grad-accum 8 --max-length 512 --eval-rows 768 \
  --lr 0.00001 --save-every 20 --attention-mode noncausal_full_attention \
  --out /content/drive/MyDrive/rl-pilot/sft --require-drive
```

Then run the same command with `--objective rl_hybrid` and `--out /content/drive/MyDrive/rl-pilot/rl_hybrid`. Preserve the initialization, data and reference-cache paths. The reference cache is generated before policy updates in evaluation mode, with a plan/data/initial-artifact fingerprint. Reuse is rejected if that fingerprint or the file hash differs. This avoids keeping a second 27B model on the GPU.

Both arms use the same frozen development and separate calibration sets. Each resulting model gets its own temperature fitted only on calibration labels. Compare accuracy, NLL, Brier, ECE and individual task regressions. Neither a lower RL training loss nor a higher training reward establishes improvement on held-out decisions.

## Related example

[Matilda's submission](https://github.com/apolinario/decision-index/pull/108) reports SFT plus RL, but does not expose an RL training recipe. Its [model card](https://huggingface.co/Maincode/matilda-jev-v1.5) also discloses benchmark-related material in earlier training and release selection using benchmark scores. Its reported improvement is not an isolated RL ablation or an independent generalization estimate. This experiment uses our own explicit objective and independent development/calibration folds.

## Full-epoch comparison

The follow-up runs the same objective for one complete pass over 92,809 eligible examples (20,601,781 nonpadding tokens): 2,901 updates, including a final update with nine examples. Both arms start from the original expanded-data SFT checkpoint, not from the pilot RL weights. [Results and learning curves](../../results/rl-decisions-full-v2/README.md).

Using the same staged data and initialization as above, run RL first and then SFT from this directory:

```bash
export PYTHONPATH=../../colab/noncausal_27b:../targeted_data_v3
for objective in rl_hybrid sft; do
  python -u train_rl.py \
    --init-artifact /content/drive/MyDrive/decision-data-pilot/targeted/selected \
    --data /content/data-v3/work/ready --train-file targeted-train.jsonl \
    --reference-cache /content/drive/MyDrive/rl-full/reference.pt \
    --objective "$objective" --full-epoch --seed 20261010 \
    --micro-batch 4 --grad-accum 8 --max-length 512 --eval-rows 768 \
    --lr 0.00001 --save-every 100 --eval-every 500 \
    --attention-mode noncausal_full_attention \
    --out "/content/drive/MyDrive/rl-full/$objective" --require-drive
done
```

The first arm computes a frozen reference cache; the second reuses it. There is no second model resident on the GPU. Diagnostics run every 500 updates and at the final update, preserving training RNG state. Temperature is fitted on a separate 768-case calibration fold. All 2,901 updates are completed; `selected` means the final calibrated artifact, not the best development checkpoint. The original small pilot remains reproducible without `--full-epoch` or `--eval-every`.

Checkpoints contain optimizer and RNG state. For recovery, preserve the original initialization, reference cache, data, schedule, and arguments, adding `--resume-artifact PATH_TO_CHECKPOINT --start-step SAVED_STEP`. Restore `training.jsonl` and `evaluation.jsonl` from that checkpoint to the output root before resuming, so logs retain the completed history. Only load trusted optimizer checkpoints. Never replace the original initialization path with the resumed policy: it identifies the frozen reference.

Our managed run used the API checkpoint transport, with a private credential held by a controller separate from the training worker. Checkpoints were uploaded and checksum-verified before training continued, with bounded retries for transient transport errors. The command above uses mounted Drive instead; authorize that mount before running it. The trainer supports `--checkpoint-spool` for an external verified-upload controller, but no account-specific credential or controller configuration is included here.

Run `python -m unittest discover -p 'test_*.py'` in this directory for objective, cache, and full-epoch coverage checks. The GPU run also checks observed token counts against the cached plan and rejects mismatched resume metadata.
