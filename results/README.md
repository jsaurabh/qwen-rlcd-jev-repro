# Frozen results

The files in this directory are measurement outputs from completed runs. They are committed so readers can inspect sample identities, predictions, probabilities, calibration metrics, runtime, memory, and configuration without rerunning large models. Personal absolute repository paths in four MLX JevBench files were replaced with the literal `$REPO`; numerical results and predictions are unchanged.

- `mlx/`: Apple M1 Pro experiments, including the publication-verification 0.5B smoke test, its earlier pre-banking-schema report, the faithful Qwen-RLCD 1.5B run, banking curricula, order-shuffle evaluations, and JevBench reports.
- `colab/nimble-9b/`: the 9B Nimble-style run and public JevBench evaluation.
- `colab/nimble-data-mix-v2/`: the data-mix experiment and public JevBench evaluation.
- `colab/autojev-27b-v4/`: the completed 27B configuration, 1,533-step log, final report, and per-case AutoJev/our-finetune/Jev comparison recovered from Google Drive.

Model weights and optimizer state are intentionally absent. They are large, tied to upstream model licenses, and belong on a model host or Drive rather than ordinary Git. The scripts regenerate them.

Accuracy and calibration values are conditional on each listed dataset. Many datasets are synthetic and related by template or contrastive construction. Compare models only within the same report/protocol.
