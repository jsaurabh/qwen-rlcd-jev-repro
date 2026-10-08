# GPU policy for the next experiment

Do not reuse the conservative compatibility settings by default. Benchmark the
fast configuration first and retain the fallback only for out-of-memory or
kernel-build failures.

## Fast configuration

- Install and verify the optimized Qwen 3.5 CUDA operators for causal convolution
  and gated delta-rule attention before training.
- Use a 1,280-token cap after asserting that every retained record fits. Reject
  the run rather than silently truncating an example.
- Disable gradient checkpointing on the 40 GB A100.
- Start with microbatch 4 and gradient accumulation 2, preserving effective
  batch size 8. Probe microbatch 8 before the full run if peak memory permits.
- Bucket examples by prompt length, with deterministic seeded ordering inside
  buckets.
- Pre-tokenize once and reuse the encoded records.
- Evaluate in padded batches instead of one record per forward pass.
- Record GPU utilization, tokens/second, examples/second, peak allocated memory,
  and step-time percentiles.

## Preflight gate

Run 20 optimizer steps with the fast configuration. Continue only when:

- loss and gradients are finite;
- peak allocated memory remains below 37 GiB;
- no example is truncated;
- candidate logits match the conservative scorer within the BF16 tolerance;
- throughput materially exceeds the previous 1.45 examples/second.

## Automatic fallback

On an out-of-memory failure, reduce microbatch 8 to 4, then 4 to 2. Enable
gradient checkpointing only if microbatch 2 still fails. If optimized extension
installation fails, record the build error and use the reference kernels; do not
silently report the fallback as the optimized run.
