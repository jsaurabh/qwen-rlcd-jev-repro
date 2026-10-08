# AutoJev-style Qwen3.5-9B LoRA

## Design

- Base: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- AutoJev prompt construction and 255 distinct single-token answer codes.
- Dedicated bias-free 255-way linear readout initialized from the corresponding LM-head rows.
- Rank-16 LoRA on attention and MLP projections; the decision readout is trained in full.
- Choice order is reshuffled during training.
- Cross-entropy supports hard and soft targets.
- Calibration uses a separate family-disjoint temperature fold and fits one scalar temperature.
- Text-only training; serving and validation retain the upstream 8,192-token contract. The pilot trains at 2,048 tokens for A100 throughput.

## Data

- 47,500 public rows reconstructed by AutoJev's pinned data builder at `--train-scale 0.5`.
- 1,920 deterministic banking counterfactual rows from `jev_gap_curriculum_v1`.
- Total: 49,420 shuffled training rows; banking share: 3.88%.
- AutoJev development, temperature, and test folds remain separate from training by source family.

## Verified preflight

- 20 optimizer steps / 160 examples.
- Peak A100 memory: 16.50 GiB.
- Wall time including calibration/evaluation: 96.43 seconds.
- 64-row development slice: 71.875% accuracy, ECE 0.1101, Brier 0.4304.
- Fitted temperature: 1.2929.
- Batch 4 test: 17.13 GiB peak. Batch 8: 18.14 GiB peak but slower on the sampled lengths, so the full pilot uses microbatch 4.

## Full pilot

The active run uses microbatch 4, gradient accumulation 8 (32 examples per optimizer step), 1,545 steps, and snapshots every 250 steps. Final calibration evaluates 512 rows from the isolated temperature fold and final selection reports 512 development rows.

The Colab runtime requires removal of Colab's incompatible preinstalled `torchaudio` and `torchao`. AutoJev also needs a one-line fix when image counts are zero so its manifest uses `0` rather than calling `max()` on an empty image split. The notebook applies all three environment fixes.

## Files

- `notebooks/autojev_9b_lora.ipynb`: reproducible Colab notebook.
- `tools/autojev_9b_lora_train.py`: trainer and artifact writer.
- `colab-results/autojev-9b-lora-preflight/autojev-9b-lora-preflight.tar.gz`: completed smoke artifact.

This reproduces AutoJev's public training mechanics with a smaller model and parameter-efficient backbone tuning. It is not the proprietary Jev architecture, and it is not the published AutoJev-27B full-weight checkpoint.
