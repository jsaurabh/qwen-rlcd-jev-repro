# Original Qwen-RLCD inference, with separately tested training

Use `faithful_rlcd.py` for the parity-first implementation. It delegates inference to the unchanged `upstream/core/engine_mlx.py`. The earlier letter-scoring demonstration remains in `mlx_jev_repro.py` for historical comparison.

## Use the exact upstream path

```sh
cd qwen-rlcd-jev-repro
.venv/bin/python faithful_rlcd.py \
  --state "A customer reports unauthorized transfers occurring right now."
```

This loads the local 4-bit Qwen2.5-1.5B model. Add `--adapter runs/faithful-v1/best` to use the validation-selected adapter and its fitted temperature. The output is the original engine's native response: `parsed_json` contains the winning values, and `field_telemetry` contains each candidate distribution. Urgency is an upstream enum (`"0"`, `"1"`, `"2"`); calculate an expected numerical score from its distribution if your application needs one. The upstream `has_calibrated_probabilities` flag is a name in that API, not a certification of calibration.

## Reproduce the experiment

```sh
.venv/bin/python faithful_rlcd.py --out runs/faithful-repeat
```

Defaults: seed 17, 80 supervised updates, 4 tickets per accumulated update, all three questions per ticket, rank-4 LoRA on query/value projections in the final four transformer blocks, learning rate 0.00005, gradient norm cap 1.0. Loss is categorical cross-entropy plus half the multiclass Brier loss. All base weights stay frozen and quantized. This experiment isolates supervised training; it does not include an RL phase. It is not TypeSafe's proprietary training method.

Inference and training keep the original JSON-field prompt, tokenizer boundary behavior, candidate token IDs, shared prefill, cache broadcasting, suffix batching, decision positions, and candidate softmax. Training retains gradients through the shared prefix cache. It rejects candidate-token collisions because upstream's fallback confidence is heuristic and not an appropriate probability-training target.

The parity test intercepts the actual input tokens to the upstream model and compares them with the training path. It also checks exact equality of native wrapper results and the upstream results, plus close equality of differentiable probabilities against the four-decimal upstream output. Parity runs both before and after training.

## Fixed data and evaluation protocol

`banking_data.py` defines an illustrative banking rubric dataset. There are 96 training cases from 24 scenarios, 24 validation cases from 12 other scenarios, and 60 test cases from 30 other scenarios. Neutral wording wrappers create related variants; these are not 180 independent test decisions. Scenario group IDs preserve those relationships. All text is synthetic; no real customer records or institution-approved labels are involved. All three splits are written and hashed before model evaluation. Training consumes only the training split.

The unchanged model (zero LoRA) and checkpoints at steps 20, 40, 60, and 80 compete on validation NLL. The selected checkpoint is evaluated on test after training; the test does not select a checkpoint. Scalar temperature is fitted separately on validation, then applied through upstream's native temperature parameter. Temperature search uses the saved four-decimal validation probabilities, with zeros clipped to 1e-12; the final calibrated test outputs are obtained by actually rerunning upstream with that temperature.

Results compare the original engine, the earlier letter scorer using the same frozen base weights, the selected supervised adapter, and that same adapter with temperature scaling. Comparing the old scorer against upstream still changes both prompting and answer representation; it does not isolate which of those two differences caused the old deficit. Comparing the faithful baseline with its supervised adapter changes weights only. Comparing that adapter with its temperature-scaled version changes temperature only.

Metrics: accuracy uses the maximum-probability candidate; Brier sums squared errors over candidates; NLL uses the gold probability clipped at 1e-12. Because the original API rounds to four decimal places, metrics inherit its finite output precision. ECE is only descriptive on this small related synthetic set. Model loading is excluded from per-case latency. Measurements are local on the M1 Pro, not API timings. No new Jev requests are part of this experiment; the earlier 12-case Jev result uses a different test set and must not be compared directly with these numbers.

## Inspect the evidence

`runs/faithful-v1/report.json` contains the measured results, parity checks, selection, memory and source hashes. `history.json` records every update and validation check. `data/manifest.json` records split hashes and scenario counts. Per-case JSON files preserve labels, distributions and timings. `best/` is the selected adapter; numbered checkpoint directories preserve other evaluated training stages. Choose a new output directory for another run rather than overwriting a finished experiment.

The prior trainer's small adapter is kept separately at `runs/banking-1.5b`. Do not mix its adapter with this prompt/scoring path when interpreting the new experiment.

## Measured result

The 80-update run completed on the M1 Pro. The validation-selected checkpoint was step 80; the test split was not used for checkpoint or temperature selection.

| Pipeline | Correct / 180 | Accuracy | NLL ↓ | Brier ↓ | ECE (5 bins) ↓ |
|---|---:|---:|---:|---:|---:|
| Earlier letter scorer, base weights | 97 | 53.9% | 0.971 | 0.622 | 0.162 |
| Original upstream path, base weights | 137 | 76.1% | 0.707 | 0.378 | 0.082 |
| Original path + selected LoRA | 149 | 82.8% | 0.496 | 0.280 | 0.083 |
| Selected LoRA + validation temperature | 149 | 82.8% | 0.488 | 0.279 | 0.079 |

Training improved overall test accuracy by 6.7 percentage points over the unchanged upstream baseline. Department routing rose from 56.7% to 86.7%; urgency fell from 83.3% to 75.0%; human-review accuracy fell slightly from 88.3% to 86.7%. The probability losses improved substantially overall. Temperature 1.035 changed probability quality slightly and did not change winners.

Parity passed before and after training. The wrapper's native answers and probabilities exactly matched the unchanged upstream engine. The differentiable training path differed from upstream's rounded probability output by at most 0.0000912. A fresh process loaded `best/` and correctly returned `fraud_review`, urgency `2`, and `escalate: true` for the saved unauthorized-wire demonstration.

Training compute took 67.25 seconds. Including periodic validation, it took 111.03 seconds; the entire run including all baseline and test evaluations took 147.17 seconds. MLX reported 1.194 GB peak allocation. Median local per-case inference was about 180 ms for the base model and 186 ms for the adapter before temperature evaluation noise.

These results come from one seed and 30 held-out synthetic scenarios represented by 60 related wording variants. They establish that the implementation now preserves the original Qwen-RLCD inference behavior and that a small supervised adapter can improve this particular rubric. They do not establish production banking accuracy, broad generalization, or calibration on real data.
