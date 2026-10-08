# JevBench comparison

This project pins the public MIT-licensed JevBench harness in `reference-jevbench` and compares the banking contrastive v3 adapter with TypeSafe Jev using the benchmark's unchanged states, questions, criteria, labels, and scoring rules.

The downloadable suite contains 48 easy, 72 standard, and 111 hard decisions. Private and imported-source items used by Benchmark Heaven are unavailable and are not reconstructed. Consequently, these results are public-split diagnostics rather than an official 534-item JevBench score.

## Smoke comparison

Run four tasks from each public tier against both systems:

```sh
./.venv/bin/python jevbench_compare.py \
  --providers both \
  --tiers easy,standard,hard \
  --limit-per-tier 4 \
  --max-tokens 2048 \
  --out runs/jevbench-smoke.json
```

The script reads `TYPESAFE_API_KEY` from the process environment or `.env`. It never writes the key to results.

## Full public comparison

```sh
./.venv/bin/python jevbench_compare.py \
  --providers both \
  --tiers easy,standard,hard \
  --max-tokens 4096 \
  --out runs/jevbench-public-full.json
```

On a 16 GB M1 Pro, start with 2,048 tokens if 4,096 causes memory pressure. Prompts longer than the configured limit are recorded as failed; they are never truncated. Run only the local model or Jev with `--providers local` or `--providers jev`.

Use `--adapter none` to evaluate the frozen base model with the identical local prompt and candidate-logit readout.
Use `--shuffle-seed 17` to permute candidates deterministically and map the resulting probabilities back to their semantic labels.

Results include accuracy, ECE, Brier score, latency, operational failures, tier breakdowns, direct prediction agreement, per-task distributions, the task-set hash, and the pinned JevBench commit. Candidate-logit probabilities are native to our local scoring method but are not the proprietary Jev architecture.

Every provider result is flushed to `<output>.partial/<provider>.jsonl` immediately. Re-running the exact command resumes by task ID. If model, adapter, tiers, provider selection, token limit, or dataset hash changes, choose a new output path.
