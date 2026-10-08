# Frozen results

These are compact outputs from the three published experiment tracks:

- `mlx/smoke-0.5b-report.json`: basic MLX mechanics and calibration smoke test.
- `mlx/nimble-1.5b-report.json`: Nimble scale-down and its answer-position collapse.
- `colab/nimble-9b/`: original 9B Nimble run and public JevBench evaluation.
- `colab/nimble-data-mix-v2/`: 9B targeted data-mix run and public JevBench evaluation.
- `colab/autojev-27b-v4/`: 27B configuration, training log, final report, and the per-case comparison with public AutoJev and Jev.

Weights and optimizer state are intentionally absent. The scripts regenerate them and write large artifacts to local storage or Google Drive.

Metrics are conditional on each listed dataset. Several datasets are synthetic or template-derived, so compare models only within the same report and protocol.
