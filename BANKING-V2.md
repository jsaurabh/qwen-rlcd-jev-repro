# Banking compositional adapter v2

This experiment keeps the public Qwen-RLCD inference mechanism: one schema/state prefill, parallel field suffixes, and candidate-token logits. It changes the banking task design and the training/evaluation discipline. It is a Qwen2.5 adapter experiment, not a reproduction of TypeSafe's proprietary Jev architecture.

## What changed

The model predicts eight atomic fields: department, suspected fraud, disputed authorization, conflicting information, policy exception, ongoing loss, customer blocked, and independent deadline pressure. Application code combines those probabilities into `needs_review` and `urgency`. This avoids asking one model output to learn a mixture of unrelated escalation rules.

The deterministic synthetic generator creates 240 training, 72 validation, and 120 test cases across 12 banking archetypes. Every row is a distinct structured state rather than the same case under several wrappers. Ambiguous suspicious-access cases use soft probability targets. These are held-out combinations from one generator, so the results test controlled compositional learning rather than real-bank or out-of-domain generalization.

Training uses rank-4 LoRA on the last four query/value projections, a 4-bit frozen Qwen2.5-1.5B base, microbatch one, four-step gradient accumulation, a 512-token cap, weighted cross-entropy plus half Brier loss, and extra cost weight for positive fraud/loss examples. A checkpoint is deployable only if validation accuracy and positive recall on suspected fraud, ongoing loss, and customer blockage remain within fixed limits of the base model. Test data never selects a checkpoint.

## Reproduce the run

```sh
cd qwen-rlcd-jev-repro

# Fast end-to-end smoke test.
.venv/bin/python faithful_v2.py \
  --steps 1 --accum 1 \
  --out runs/compositional-v2-smoke-repeat

# Full measured run on the M1 Pro.
.venv/bin/python faithful_v2.py \
  --steps 100 --accum 4 --lr 0.00004 \
  --out runs/compositional-v2-cost-aware-repeat
```

Run the saved adapter on one structured case:

```sh
.venv/bin/python faithful_v2.py \
  --adapter runs/compositional-v2-cost-aware/best \
  --state '{"customer_message":"I did not authorize the pending wire and funds may leave within hours.","observed_facts":{"account_secured":false,"transaction_pending":true,"customer_access_available":true},"policy_context":{"deadline":"within hours","exception_requested":false,"named_rule":"none"}}'
```

The response contains the public engine's raw per-field decisions and candidate probabilities plus code-derived `needs_review` and `urgency` distributions.

## Measured result

The selected checkpoint was step 100. The training loop used 163.5 seconds of measured compute; the complete training/evaluation phase took 332.9 seconds with the model already local. Median warm per-case evaluation was about 342 ms for eight parallel fields. The 4-bit model directory is 839 MB, the complete run is 2.8 MB, and the selected adapter is 306 KB. Total process peak memory was not measured; on a 16 GiB M1 Pro, keep microbatch one and shorten the context or reduce adapted layers first if memory pressure appears.

| Frozen synthetic test metric | Base | Selected adapter |
|---|---:|---:|
| Atomic accuracy (960 decisions) | 76.98% | 97.60% |
| Weighted NLL | 0.756 | 0.278 |
| Suspected-fraud recall | 66.67% | 96.67% |
| Ongoing-loss recall | 100% | 100% |
| Customer-blocked recall | 100% | 100% |
| Derived `needs_review` accuracy | 93.33% | 100% |
| Derived `urgency` accuracy | 40.83% | 100% |
| Operational cost per case | 0.933 | 0.050 |

Exact configuration, per-field metrics, timing, data hashes, checkpoints, and predictions are in `runs/compositional-v2-cost-aware/report.json` and its sibling files.

## Scaling next

Before increasing model size or LoRA rank, replace part of the synthetic test with reviewed, de-identified institutional cases and hold out whole products, channels, and incident families. Fit decision thresholds on a separate calibration set using the bank's actual false-negative and false-positive costs. Run several seeds and adversarial minimal pairs, including one-fact changes such as secured versus still-active access.

[Bespoke Nimble](https://github.com/bespokelabsai/nimble), pinned locally at commit `f136b3f75721fda4ea961f73993cc50b08488835` under `reference-nimble/`, provides a strong public recipe for that next dataset iteration. Its most relevant controls are: base/counterfactual pairs that change one decisive fact; keeping every sibling from one source family in the same split; deletion checks that verify neither evidence sentence alone reveals the answer; candidate-order shuffling to measure letter-position bias; a saved contract tying the adapter to its base revision, tokenizer, and prompt implementation; and external human-labeled benchmarks. Its published 9B training recipe is not suitable unchanged for this 16 GiB M1 Pro: it uses unquantized BF16 Qwen3.5-9B, rank-16 LoRA, 2,676 examples, and GPU training. We should adopt its data and evaluation controls while retaining the local 4-bit Qwen2.5-1.5B, rank-4 MLX setup.

Once that evaluation is stable, expand to more distinct scenarios, then adapt 8 layers or raise LoRA rank to 8. Keep microbatch one on a 16 GiB M1 Pro and lower the learning rate when increasing adapter capacity. A larger Qwen model will increase latency and memory without fixing label design, leakage, or calibration on its own.
