# Frozen banking comparison: Qwen-RLCD, our LoRA, and Jev

Run completed September 21, 2026 on the same frozen test split used by `runs/faithful-v1/report.json`. The split contains 60 synthetic banking messages representing 30 underlying situations, with three labeled decisions per message: department, urgency, and human review. The two messages per situation are related wording variants, not independent samples.

| Pipeline | Correct / 180 | Accuracy | NLL ↓ | Brier ↓ | ECE, 5 bins ↓ | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| Original community Qwen-RLCD | 137 | 76.1% | 0.707 | 0.378 | 0.082 | 178.5 ms |
| Our upstream-compatible LoRA | 149 | 82.8% | 0.496 | 0.280 | 0.083 | 179.1 ms |
| Official Jev API (`jev-1.13.0`) | **178** | **98.9%** | **0.071** | **0.027** | **0.047** | **106.6 ms** |

## Per-field accuracy

| Pipeline | Department | Urgency | Human review |
|---|---:|---:|---:|
| Original community Qwen-RLCD | 56.7% | 83.3% | 88.3% |
| Our upstream-compatible LoRA | 86.7% | 75.0% | 86.7% |
| Official Jev API | **100%** | **96.7%** | **100%** |

Jev's two incorrect decisions were the two wording variants of the same underlying situation: a settled unfamiliar purchase where ongoing loss was ruled out. It selected urgency 0 while the authored rubric label was urgency 1. At the scenario-group level, Jev therefore missed one of 30 situations. The original Qwen-RLCD pipeline made at least one error in 19 groups; our adapter did so in 15 groups.

## What was held constant

- All three pipelines used exactly the same 60 state strings, labels, candidate meanings, and dataset SHA-256: `bffba973c8e06101d12002aa7b94020cff48b4383932e6d2a990f1fa98f9289a`.
- Original and adapted local models both used the same local 4-bit Qwen2.5-1.5B checkpoint and the unchanged `upstream/core/engine_mlx.py` inference path. The adapted pipeline changed only the validation-selected LoRA weights.
- Jev received equivalent Choice, Score, and Noul questions in one request per state. All 60 requests succeeded without retry. `jev-latest` resolved to `jev-1.13.0` for every response.
- Jev reported 28,048 input tokens and 4,500 output tokens across the run. These are provider-reported usage counts, not a verified invoice.
- Local latency excludes model loading and shader warmup. Jev latency includes HTTP, network, and service time. Calls were sequential and each case was timed once. These are application measurements, not controlled hardware-throughput measurements.

The local Qwen-RLCD engine and Jev use different native prompt and scoring implementations, although the semantic state and questions are aligned. The result compares usable pipelines rather than isolating architecture alone.

## Interpretation and limits

The faithful LoRA improved the original local pipeline by 12 correct decisions and reduced NLL and Brier loss substantially. Its department routing improved sharply, but urgency and review accuracy were slightly worse than the original baseline. Jev was substantially stronger on every aggregate metric and every field in this pilot.

The examples and gold labels are hand-authored for an illustrative routing policy. They are synthetically phrased, close to the training domain, and contain related variants. No real customer data, institution-approved policy, or production outcomes were used. The small ECE values do not establish general calibration. This run supports a narrow conclusion: on this fixed synthetic banking rubric, Jev greatly outperformed both local pipelines, while our faithful LoRA improved overall performance over the original community Qwen-RLCD implementation.

## Files and reproduction

Raw per-case distributions, labels, latency, Jev responses, and summaries are under `runs/faithful-comparison-v1/`. The benchmark script never writes the API key or request headers to results.

```sh
cd qwen-rlcd-jev-repro

# Local runs only
.venv/bin/python compare_faithful.py --backend original --out runs/comparison-repeat
.venv/bin/python compare_faithful.py --backend ours --out runs/comparison-repeat

# Makes 60 Jev API requests using TYPESAFE_API_KEY from .env
.venv/bin/python compare_faithful.py --backend jev --out runs/comparison-repeat
```

Use a new output directory to preserve the completed run. The script refuses dataset-hash mismatches when combining result files.
