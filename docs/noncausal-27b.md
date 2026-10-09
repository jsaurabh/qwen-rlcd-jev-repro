# Frozen noncausal 27B experiment

This is the independent LoRA/readout model submitted in [Decision Index PR114](https://github.com/apolinario/decision-index/pull/114). It is separate from both the earlier banking 27B run and the official Perplexity model.

## Run the published checkpoint

Use Linux/Python3.13 and a CUDA GPU; the evaluated hardware was one RTX PRO6000 Blackwell96GB. Model weights are hosted separately to keep this repository small.

```bash
cd colab/noncausal_27b
python -m pip install -r requirements.txt
python download_artifact.py
DECISION_ARTIFACT="$PWD/artifact" python -u serve_checkpoint.py
```

The server exposes `GET /health` and `POST /v1/systemone` on `127.0.0.1:8765`, using model name `our-noncausal-27b-step1533`. The portable server changes only artifact/provenance paths relative to the evaluated original. It requires no Jev key. The base Qwen weights download separately at the pinned revision.

## Training configuration

- Qwen/Qwen3.8-27B, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Rank16 LoRA, alpha32, dropout0.05 plus a255-way last-token readout;80,997,376 trainable parameters.
- Full SDPA layers use noncausal attention; linear-attention recurrence is preserved.
- 48,847 training rows,1533steps, microbatch4, accumulation8, learning rate5e-5, seed20260922,512-token training context.
- Development and temperature calibration each use512rows; saved temperature1.0862525693910237.
- Optimizer/RNG checkpoints support exact resume when present; inference publication omits optimizer state.

With the prepared training/dev/temperature JSONL files:

```bash
cd colab/noncausal_27b
python -u autojev_lora_train.py \
  --data /path/to/prepared-data \
  --attention-mode noncausal_full_attention \
  --steps 1533 --micro-batch 4 --grad-accum 8 \
  --max-length 512 --eval-rows 512 --save-every 100 \
  --out /path/to/persistent-checkpoints
```

`prepare_training_data.py` preserves the original preparation script and its `/content` paths. It expects the archived `autojev-data-v2.tar.gz`, gap training file and upstream processor. It selected a deterministic two-thirds text subset per suite from that archive, added synthetic banking gap examples, and filtered to480tokens before option shuffling. The committed training manifest records exact hashes. Rebuilding upstream data today is not guaranteed to reproduce the archived input byte-for-byte; the archived corpus itself is not redistributed here.

## Evaluation

Official kit revision `9eb2dbe2a358004c8782c66e40a83ac07b953fec`, full public0.3 suite, HTTP engine. Serve the model, then in the kit environment:

```bash
python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8765 \
  --option model=our-noncausal-27b-step1533 \
  --option timeout=3600 --suite-dir /path/to/suite-0.3 \
  --out runs/noncausal-step1533 --compact
```

Frozen capacity:255options,8192tokens per complete question branch,four questions per forward batch,choice/noul,no truncation. Saved training temperature; no benchmark tuning. The run used the PyTorch causal-convolution fallback.

Public index57.70, raw68.26;140178scoreable requests (140148supported,30unsupported,0errors). Latency median105.9ms, mean383.2ms,p951565.3ms. These do not establish the maintainer latency gate or private leaderboard ranking. The runner's140620total also includes442excluded requests.

The complete results include reused predictions from the same checkpoint/protocol. Aggregate JSON is in `results/decision-index-0.3/noncausal-27b/`; compact predictions are on Hugging Face. No benchmark prompts are published.

## Data limitations and follow-up

Training/public-benchmark overlap is not fully audited. Namespaced family separation among our train/dev/calibration splits does not establish benchmark independence. The next experiment starts by auditing the exact archived rows, then compares a cleaned control with targeted additional data at equal training-token budgets. No improvement is claimed before that comparison.
