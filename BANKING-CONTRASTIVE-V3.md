# Banking contrastive v3

This experiment applies the transferable parts of Bespoke Nimble's recipe to a banking-only curriculum that fits Qwen2.5-1.5B on a 16 GiB M1 Pro. It is a successful controlled experiment, not evidence of production banking accuracy.

## Dataset and controls

The deterministic dataset contains eight atomic banking decisions: suspected fraud, disputed authorization, conflicting information, policy exception, ongoing loss, customer blocked, department routing, and deadline pressure.

Every source family is a two-record minimal pair. Both records share one explicit policy and differ in one decisive case fact; that fact flips the reference answer. Each record carries a mechanical evidence certificate asserting that removing either the policy or the case fact makes the answer unknown. Whole families are isolated by split:

| Split | Records | Complete pairs |
|---|---:|---:|
| Training | 512 | 256 |
| Validation | 128 | 64 |
| Test | 128 | 64 |

Training candidate order is seeded independently per record. Validation and test are each evaluated twice: canonical order and a separate seeded permutation. Pair-aware batches keep four complete pairs in each effective batch of eight. The final test is not used for checkpoint selection.

The deployment gate requires a checkpoint to beat the base model's worst-order individual accuracy and worst-order complete-pair accuracy. It also rejects a checkpoint if more than 80% of predictions occupy one candidate position.

## Model and training

- Quantized Qwen2.5-1.5B base
- Rank-8 LoRA over all attention and MLP projections in the final eight blocks
- 64 optimizer steps, exactly one pass over 512 records
- Microbatch one, gradient accumulation eight
- Candidate-only cross-entropy, gradient norm clipped to 1
- Learning rate `5e-5`, six-step warmup, linear decay to zero
- Maximum prompt length 768; no truncation

```sh
cd qwen-rlcd-jev-repro
.venv/bin/python banking_contrastive_mlx.py \
  --out runs/banking-contrastive-v3
```

Training compute took 172.7 seconds. The complete run, including repeated validation and final tests, took 462.6 seconds. The selected adapter is 10 MB. Reloading it in a fresh model produced exactly the saved probabilities on the checked records.

## Results

Checkpoint 64 passed the validation gate and was selected without consulting test labels.

| Untouched synthetic test | Canonical order | Permuted order |
|---|---:|---:|
| Individual accuracy | 100% (128/128) | 100% (128/128) |
| Complete-pair accuracy | 100% (64/64) | 100% (64/64) |
| NLL | 0.000057 | 0.000074 |
| ECE, 10 bins | 0.000057 | 0.000074 |

Selected positions remain distributed across A, B, and C in both evaluations. This rules out the candidate-position collapse seen in the broad-domain scaled Nimble experiment.

## Interpretation

The result establishes that a 1.5B model can learn candidate-order-robust minimal-pair banking rules when the curriculum is narrow and explicit. It does not establish general banking understanding. Policies are included verbatim, the sentence structures repeat across splits, labels are deterministic, and no example was written or reviewed by a banking professional.

The next evaluation should preserve the schema but replace the test split with independently authored and de-identified cases. Hold out entire policy families, products, channels, institutions, and linguistic styles. Add unknown/insufficient-evidence outcomes so missing evidence is not forced into false. Validate probability thresholds against the institution's false-negative costs before treating the very sharp synthetic probabilities as calibrated.

## Files

- `banking_contrastive_data.py`: deterministic minimal-pair generator and hashes
- `banking_contrastive_mlx.py`: pair-aware trainer, checkpoint gate, and two-order evaluation
- `runs/banking-contrastive-v3/schema_config.json`: model, prompt, LoRA, data, and selection contract
- `runs/banking-contrastive-v3/report.json`: final aggregate report
- `runs/banking-contrastive-v3/test.json`: per-record probabilities for both candidate orders
- `runs/banking-contrastive-v3/best/`: selected MLX adapter
