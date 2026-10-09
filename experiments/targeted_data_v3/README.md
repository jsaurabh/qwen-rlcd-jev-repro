# Matched training-data pilot

This experiment compares the existing decision-data mixture against an expanded mixture, starting from the same Qwen base, language-model readout, LoRA initialization, and seed. The completed pilot improved shared development accuracy from **81.12% to 86.85%**. See [the full comparison](../../results/targeted-data-v3/README.md), including regressions and limits. This has not been evaluated as a new leaderboard submission.

## Data and compute

| | Control | Expanded mixture |
|---|---:|---:|
| Eligible training examples | 47,852 | 92,809 |
| Available nonpadding tokens | 12,820,934 | 20,601,781 |
| Training budget | approximately 2 million tokens | approximately 2 million tokens |

Both arms use the same 768 development examples and a separate 768-example calibration set. Selection is approximately balanced across task suites. A seeded shuffle selects a single-pass training prefix; the last example that would exceed the budget is omitted. Thus budgets may differ by fewer than 480 tokens. This is a compute-matched pilot, not a full epoch over either dataset.

The expanded pool combines existing decisions, 23,971 ACOS quadruple-verification decisions, 5,392 ACOS sentiment decisions, 17,970 ESCI relevance decisions, and 16,000 verified Python-expression decisions. The original proposed allocation was adjusted to usable unique examples; NuminaMath and Tulu were collected but not converted or used, and xLAM was unavailable. The code tasks are short procedural arithmetic exercises, not a general reasoning distillation dataset.

`token-manifest.json` records exact counts by source, token totals, and file hashes. Raw downloaded corpora and benchmark records are not distributed here. ACOS is an author research release without an explicit license file in the pinned source; we do not claim redistribution permission. Review source terms before independent reuse.

## Reproduction stages

The scripts preserve the Colab filesystem layout used in this experiment. Stage existing `train.jsonl`, `dev.jsonl`, and `temperature.jsonl` under `/content/data-v3/work/original`, and your private benchmark `*rows.jsonl.gz` under `/content/data-v3/work/suite-0.3`.

1. Run the earlier `audit_overlap.py` to produce the preliminary `work/audit` inputs.
2. Install `datasketch`, `datasets`, `huggingface-hub`, `pyarrow`, and `pandas`; run `collect_sources.py`. It records revisions, hashes and download/access failures. Its extra source collection does not imply training inclusion.
3. Run `build_candidates.py`. ACOS splits by review family; ESCI uses only US training examples with query-family splits and rejects products crossing folds; code templates are separated across training, development and calibration.
4. Run `screen_candidates.py`. It screens both arms against benchmark documents and shared held-out documents, and removes repeated state/question pairs.
5. Run `prepare_tokens.py` with `transformers==5.17.0`. It uses the pinned published tokenizer and chat template, validates targets, freezes option order, rejects inputs above 480 tokens and selects the shared held-out sets. It does not load model weights.
6. Run `test_code_labels.py` (1,200 formula checks), `test_token_plan.py` and `test_checkpoint_mailbox.py`. Use `train_token_budget.py` with the runtime and dependencies from `../../colab/noncausal_27b` on `PYTHONPATH`.

Example training invocation (run once for each arm with separate output directories):

```bash
PYTHONPATH=../../colab/noncausal_27b python train_token_budget.py \
  --data /content/data-v3/work/ready \
  --train-file control-train.jsonl \
  --token-budget 2000000 --seed 20261009 \
  --micro-batch 4 --grad-accum 8 --max-length 512 \
  --eval-rows 768 --lr 0.00005 --save-every 50 \
  --attention-mode noncausal_full_attention \
  --out /content/drive/MyDrive/decision-data-pilot/control --require-drive
```

For the second arm, change the training filename to `targeted-train.jsonl` and output directory to `targeted`. These commands require an already mounted Drive. Our automated run instead uses a separate credential-holding upload controller and `--checkpoint-spool`; that flag requires a running controller and must not be used on its own.

The runner checks actual processor token counts against preparation counts on every batch. Classification loss averages decisions, including partial final batches; longer examples do not receive extra loss weight. Learning rate uses 5% token warmup followed by linear decay. Checkpoints contain optimizer and RNG state plus plan hash, step, examples and tokens consumed. Exact resume requires the original data, seed, batch configuration and token budget.

## Screening limitations and environment

The stronger screen removed 11 control and 366 expanded-mixture training rows matching benchmark documents, before token filtering. This supersedes the preliminary whole-state-only audit. It compares normalized whole states and nested string fields, using exact hashes and MinHash candidates confirmed by five-word-shingle Jaccard >= 0.8. Near comparisons use the first 4,096 words. Shared documents are conservatively removed even when questions differ. Passing this screen does not prove absence of semantic or pretraining contamination.

On Colab's RTX PRO 6000 Blackwell, a short test of batch sizes 4/8/16 favored 4; both arms therefore use batch 4 with accumulation 8. The environment needed removal of an unused preinstalled `torchao==0.10.0`, which conflicted with the pinned PEFT version. Isolated worker environments must preserve Colab's `LD_LIBRARY_PATH` to find CUDA. `causal_conv1d` was not installed; both arms use the same reference fallback.

Do not select recipes or fit temperature against the public benchmark. Development comparisons and calibration use their own fixed folds; benchmark evaluation comes after recipe selection.
