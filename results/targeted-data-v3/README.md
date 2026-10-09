# Targeted-data pilot results

The expanded data mixture improved accuracy on the fixed 768-example development set by **5.73 percentage points**. Both runs started from the same base/readout/LoRA initialization procedure and seed, with matching hyperparameters and identical development/calibration file hashes. Each consumed approximately two million nonpadding input tokens.

| Calibrated metric | Control | Expanded mixture |
|---|---:|---:|
| Accuracy | 0.8112 | 0.8685 |
| NLL | 0.4768 | 0.3419 |
| BRIER | 0.2484 | 0.1843 |
| ECE | 0.0364 | 0.0234 |

Higher accuracy is better; lower NLL, Brier and ECE are better. Each temperature was fitted separately on the same 768-example calibration fold.

## Task-level accuracy

| Task | Examples | Control | Expanded mixture | Change (pp) |
|---|---:|---:|---:|---:|
| acos-quad-verification | 55 | 83.64% | 96.36% | +12.73 |
| acos-sentiment | 55 | 98.18% | 96.36% | -1.82 |
| aegis2 | 55 | 81.82% | 87.27% | +5.45 |
| boolq | 55 | 94.55% | 94.55% | +0.00 |
| civil_comments | 55 | 83.64% | 87.27% | +3.64 |
| esci-relevance | 55 | 54.55% | 67.27% | +12.73 |
| massive-de-DE | 55 | 92.73% | 89.09% | -3.64 |
| massive-en-US | 55 | 92.73% | 89.09% | -3.64 |
| multinli | 55 | 89.09% | 94.55% | +5.45 |
| paws | 55 | 94.55% | 92.73% | -1.82 |
| pubmedqa | 55 | 69.09% | 76.36% | +7.27 |
| squad2 | 55 | 92.73% | 87.27% | -5.45 |
| verified-code | 54 | 24.07% | 74.07% | +50.00 |
| vitaminc-dev | 54 | 83.33% | 83.33% | +0.00 |

The largest gains were on the held-out synthetic code templates, ACOS quadruple verification, and ESCI relevance. Several established tasks regressed, including SQuAD2, MASSIVE English/German and PAWS. The expanded mixture is therefore a promising candidate for further validation, not an unconditional replacement for the published model.

## Compute and reproducibility

- Control: 234 optimizer updates, 7,487 examples, 1,999,906 input tokens; 45.2 minutes including evaluation and periodic checkpoint uploads before final artifact upload.
- Expanded: 283 optimizer updates, 9,027 examples, 1,999,882 input tokens; 49.3 minutes on the same basis.
- Token budgets differ by only 24 tokens. More, shorter examples cause more updates in the expanded arm; this is token-compute matching, not update-count matching.
- One RTX PRO 6000 Blackwell (96 GB); peak allocated memory about 52.3 GiB. Microbatch 4, accumulation 8; context 512, input filter 480.
- All final model and optimizer artifacts were verified on private Drive; the GPU was released. Raw training inputs and model weights are not included in this repository.

## Interpretation limits

These are internal development results, **not Decision Index scores** and not a direct comparison against Jev or Perplexity Decider. The experiment uses one seed and only 54–55 development cases per task. Aggregate reports cannot establish paired statistical significance. The data mixture and held-out construction favor coverage of the included tasks; no claim of general improvement follows from the overall average alone.

The code tasks are simple generated Python-expression exercises. Benchmark-overlap screening is conservative but does not prove semantic or pretraining independence. Calibration uses held-out labels; it is not a guarantee under distribution shift. End-to-end serving latency was not measured in this pilot.

## Files

- `control-report.json`, `targeted-report.json`: complete aggregate metrics, including raw/calibrated task results and reliability bins.
- `*-run_config.json`: model revision, configuration, plan and data hashes.
- `*-training.jsonl`: per-step loss, token counts, learning rates, time and allocated memory.
- `comparison.json`: machine-readable deltas.

The [recipe and source provenance](../../experiments/targeted_data_v3/README.md) reproduce data preparation and training. The existing published model and leaderboard submission remain unchanged.
