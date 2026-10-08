# Banking v4 result

V4 was trained from the frozen Qwen2.5-1.5B base on 768 synthetic banking decisions. JevBench was excluded from training, validation, calibration, and checkpoint selection.

## Outcome

| Evaluation | V3 | V4 | V4 shuffled |
|---|---:|---:|---:|
| Public JevBench overall | 136/231 | 125/231 | 122/231 |
| Easy | 45/48 | 47/48 | 45/48 |
| Standard | 45/72 | 48/72 | 42/72 |
| Hard | 46/111 | 30/111 | 35/111 |
| ECE | 0.396 | 0.167 | 0.167 |
| Brier | 0.797 | 0.564 | 0.593 |

V4 is better calibrated and stronger on easy and standard tasks, but it regresses hard reasoning enough that it must not replace V3 as the best accuracy checkpoint.

## Family comparison

| Family | N | V3 | V4 | V4 shuffled |
|---|---:|---:|---:|---:|
| adequacy | 12 | 6/12 | 5/12 | 4/12 |
| adversarial | 6 | 3/6 | 3/6 | 2/6 |
| ambiguous | 7 | 3/7 | 1/7 | 3/7 |
| extraction | 24 | 19/24 | 20/24 | 20/24 |
| fact | 12 | 11/12 | 12/12 | 10/12 |
| intent | 24 | 21/24 | 21/24 | 16/24 |
| judge_hard | 17 | 7/17 | 7/17 | 7/17 |
| long_policy | 19 | 7/19 | 4/19 | 4/19 |
| multi_hop | 18 | 4/18 | 3/18 | 4/18 |
| ordinal | 12 | 7/12 | 10/12 | 9/12 |
| policy | 12 | 6/12 | 7/12 | 9/12 |
| probability | 10 | 7/10 | 2/10 | 3/10 |
| routing | 12 | 8/12 | 8/12 | 7/12 |
| routing_hard | 5 | 4/5 | 1/5 | 2/5 |
| temporal_numeric | 15 | 7/15 | 5/15 | 4/15 |
| tool_selection | 12 | 12/12 | 12/12 | 12/12 |
| tradeoff | 6 | 4/6 | 2/6 | 2/6 |
| trap | 8 | 0/8 | 2/8 | 4/8 |

## Decision

Keep V3 as the accuracy baseline. Keep V4 as a calibration and curriculum experiment. The next run should use harder, reviewed or teacher-generated policy cases with verified rationales rather than templated synthetic complexity. Distill intermediate rule applications into training targets or use a larger teacher to create counterfactual long-policy pairs, while keeping JevBench public cases excluded.

V4 artifacts:

- `runs/banking-v4/best/adapters.safetensors`
- `runs/banking-v4/best/calibration.json`
- `runs/banking-v4/report.json`
- `runs/jevbench-public-v4-v1.json`
- `runs/jevbench-public-v4-shuffled-v1.json`
