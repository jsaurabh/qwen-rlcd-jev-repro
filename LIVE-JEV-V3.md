# Live Jev comparison for banking contrastive v3

Eight fresh, hand-authored deterministic banking cases were scored by the selected local v3 adapter and the live `jev-latest` API. Both systems received the same structured state and one typed question per request. Reference labels were retained by the evaluator and were not included in either model's input. The API key was read from `.env` and was not saved.

| Case | Reference | Local v3 | Jev |
|---|---|---|---|
| Suspected fraud | `true` | `true` · 99.99% | `true` · 96% |
| Disputed authorization | `false` | `false` · ~100% | `false` · 97% |
| Conflicting records | `true` | `true` · 99.83% | `true` · 98% |
| Policy exception | `false` | `false` · ~100% | `false` · 95% |
| Ongoing loss | `true` | `true` · 99.98% | `true` · 97% |
| Customer blocked | `false` | `false` · ~100% | `false` · 97% |
| Department | `fraud_review` | `fraud_review` · 99.998% | `fraud_review` · 100% |
| Deadline tomorrow afternoon | `Prompt` | **`Immediate` · 99.98%** | `Prompt` · 97% |

| Aggregate | Local v3 | Jev |
|---|---:|---:|
| Accuracy | 87.5% (7/8) | 100% (8/8) |
| Mean NLL | 1.121 | 0.029 |
| Mean multiclass Brier | 0.250 | 0.002 |
| Median observed request time | 203.6 ms | 157.9 ms |

The timing is illustrative rather than a controlled service benchmark. Local measurements used a warm model on this M1 Pro. Jev measurements include network and service time for eight sequential single-question requests.

The deadline error exposes the most important gap in the synthetic curriculum. The adapter saw `Routine` examples with no deadline within a week and `Immediate` examples with deadlines within two hours, but it did not see a `Prompt` training target. It therefore learned a sharp endpoint distinction without learning the full three-level rubric. Its extreme 99.98% probability on the wrong answer also shows that its synthetic-test probabilities are not calibrated for novel cases.

The exact redacted inputs, probability distributions, model identifiers, and timings are stored in `runs/live-jev-v3.json`. The reproducible evaluator is `live_jev_v3_eval.py`.
