# Hugging Face adapter vs Jev live banking comparison

## Result

The published `jsaurabh/qwen3.5-9b-jev-data-mix-v2` adapter and `jev-latest` were evaluated on the same 12 labeled banking cases and the same three typed judgments per case.

| Backend | Overall | Department | Urgency | Escalate | NLL | Brier | ECE (5 bins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B data-mix v2 | 35/36 (97.22%) | 12/12 | 12/12 | 11/12 | 0.0752 | 0.0431 | 0.0242 |
| Jev (`jev-latest`) | 36/36 (100%) | 12/12 | 12/12 | 12/12 | 0.0319 | 0.00494 | 0.0306 |

The adapter's only error was the escalation judgment for: “A recognized transfer is overdue and preventing a purchase. There is no fraud or policy exception.” The expected answer was no escalation. The adapter assigned `P(no)=0.2227` and `P(yes)=0.7773`; Jev assigned `P(no)=0.89` and `P(yes)=0.11`.

This is a small authored smoke evaluation, so it demonstrates that the end-to-end comparison works but is not a stable estimate of production quality.

## Reproduction

The Qwen adapter was scored on an A100 with the frozen Nimble candidate-logit readout. The Jev calls ran locally so the TypeSafe API key was never copied to Colab. `mlx_jev_repro.py` now accepts a local prediction file and produces the side-by-side comparison:

```bash
.venv/bin/python mlx_jev_repro.py \
  --local-results runs/hf-adapter-jev-live/local-qwen.json \
  --compare-out runs/hf-adapter-jev-live/comparison.json \
  --jev-model jev-latest
```

The comparison JSON contains every input, label, full probability distribution from both systems, aggregate metrics, and model provenance.
