# Banking comparison: Jev, community Qwen-RLCD, and our implementation

Run completed September 21, 2026 on the M1 Pro. Official Jev matched all 36 hand-authored rubric decisions on this small pilot. The original community Qwen-RLCD implementation outperformed our current scorer. This is a 12-case synthetic pilot, not a production banking benchmark or proof of calibration.

| Implementation | Correct / 36 | Accuracy | Brier loss ↓ | NLL ↓ | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Official Jev API | 36 | 100.0% | 0.008 | 0.040 | 102.9 | 210.7 |
| Original community Qwen-RLCD | 29 | 80.6% | 0.294 | 0.501 | 179.3 | 286.2 |
| Our candidate scorer, base model | 20 | 55.6% | 0.600 | 0.932 | 200.1 | 205.5 |
| Our LoRA scorer + temperature 2 | 16 | 44.4% | 0.565 | 0.857 | 200.7 | 202.1 |

## Per-field accuracy

| Implementation | Department | Urgency | Human review |
|---|---:|---:|---:|
| Official Jev API | 100.0% | 100.0% | 100.0% |
| Original community Qwen-RLCD | 58.3% | 83.3% | 100.0% |
| Our candidate scorer, base model | 58.3% | 58.3% | 50.0% |
| Our LoRA scorer + temperature 2 | 41.7% | 41.7% | 50.0% |

## What was compared

- Official TypeSafe endpoint `https://api.typesafe.ai/v1/systemone`, requested `jev-latest`, returned **jev-1.13.0** for every request. Twelve requests completed successfully, without retries. Reported usage totaled 5,507 input tokens and 900 output tokens; this is usage, not an invoice or verified dollar charge.
- The original `upstream/core/engine_mlx.py` remained unchanged. It used our downloaded **Qwen2.5-1.5B-Instruct-4bit** checkpoint, the same weights used by our base scorer. This is the community `harshatheg/Qwen-2.5-1B-RLCD` release, not an official Qwen training method.
- Our base scorer uses the same weights with the letter-candidate prompt/readout we implemented.
- Our adapted scorer additionally loads `runs/banking-1.5b`, including temperature 2 fitted earlier on its separate synthetic validation set. These twelve pilot cases were not used to fit that temperature. They are new wording but close to the training domain and templates.
- Every implementation received the same states and semantic banking rubric. Each used its native interface: upstream enums/boolean, our letter-candidate schema, and Jev Choice/Score/Noul. Exact token strings and internal computation differ. This is a comparison of the implemented pipelines, not an isolated architecture experiment.
- Upstream candidate-token collisions were explicitly checked and absent for these choices. Its collision fallback constructs heuristic confidence; this benchmark stops rather than using that fallback in probability metrics.
- API response distributions and local distributions were validated. Tiny rounding discrepancies were renormalized only within a 0.002 sum tolerance. All 48 pipeline case evaluations succeeded.

## How to interpret the numbers

Accuracy is the highest-probability category for each of three fields (including the highest-probability urgency level and a 0.5 binary threshold). Brier is the sum of squared probability errors against each one-hot label, averaged across fields/cases. NLL is negative log probability of the gold category (clipped at 1e-12). Lower Brier and NLL are better. Provider-specific `confidence` fields are not compared: Jev Score confidence has a different definition. Five-bin ECE is saved in JSON, but 36 correlated field decisions on 12 easy cases cannot establish calibration.

Local latency excludes model loading and a separate warmup and includes a fresh shared-prefix prefill per case. Jev latency includes HTTP, network and service time, including the first connection. Requests were sequential per backend, each case measured once. The online API pass and local benchmark overlapped in wall time, but the API pass performed no model inference locally. p95 is interpolated from only twelve samples. These are observed application timings, not controlled model-compute speed measurements.

Gold labels were authored for this illustrative rubric. No real customer data or institution-approved policy was used. No statistical or out-of-distribution claim follows from this pilot. Jev was not used as the source of the gold labels.

## Example errors

For a routine on-time transfer inquiry, Jev selected payments, urgency 0, and no review. Upstream selected account support but got urgency and review right. Both of our variants selected account support, urgency 1, and review.

For unauthorized transfers occurring now, Jev and upstream matched fraud review, urgency 2, and review. Our base scorer selected urgency 1; the adapter also routed to account support. The larger model alone has not fixed our prompting/readout weaknesses, and the 16-update adapter does not improve this pilot.

## Repeat the comparison

```sh
cd qwen-rlcd-jev-repro
.venv/bin/python compare.py --backend local --out runs/comparison-repeat
.venv/bin/python compare.py --backend jev --out runs/comparison-repeat
```

The second command reads `TYPESAFE_API_KEY` from the environment or the project `.env` and makes twelve billable API requests. The key is not printed or written to results. `--backend upstream`, `--backend ours_base`, and `--backend ours_adapter` run just one local variant. A repeated command reruns its backend; choose a new output directory to preserve old results. `jev-latest` may resolve to a newer model later; every raw response records the returned version.

`runs/comparison-banking/cases.json` contains the exact cases, labels and schema. The four backend JSON files preserve per-case probability distributions, raw answers, latency and metadata; `summary.json` holds aggregate results. `benchmark_provenance.json` records source revisions and the dataset hash.

## Sources

- [TypeSafe introduction](https://docs.typesafe.ai/introduction): the mixed typed-decision interface.
- [TypeSafe Score documentation](https://docs.typesafe.ai/primitives/score): rubric levels, probabilities and score semantics.
- [Official TypeSafe Python SDK source](https://github.com/typesafe-ai/typesafe-sdk-python): verified request/response and authentication shapes. The benchmark calls HTTP directly using the already-installed httpx.
- [Community Qwen-RLCD release](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD): the original implementation being compared.
