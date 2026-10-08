# Public JevBench evaluation

Evaluated on the 231 publicly downloadable JevBench v1.2 tasks at pinned commit `5e95f23cbb7be098a9061fea924c4421620ab1a5`. This is a public-split diagnostic, not the official 534-task leaderboard score. The private and imported-source items are unavailable.

| System | Overall | Easy | Standard | Hard | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B base | 118/231 (51.1%) | 43/48 | 43/72 | 32/111 | 0.362 | 0.774 |
| Banking contrastive v3 | 136/231 (58.9%) | 45/48 | 45/72 | 46/111 | 0.396 | 0.797 |
| Banking v3, shuffled candidates | 129/231 (55.8%) | 46/48 | 41/72 | 42/111 | 0.412 | 0.835 |
| Banking v4 | 125/231 (54.1%) | 47/48 | 48/72 | 30/111 | 0.167 | 0.564 |
| Banking v4, shuffled candidates | 122/231 (52.8%) | 45/48 | 42/72 | 35/111 | 0.167 | 0.593 |
| Jev 1.13.0 | 201/231 (87.0%) | 48/48 | 71/72 | 82/111 | 0.055 | 0.181 |

The adapter adds 18 correct decisions over the base, including 14 on the hard tier. Its probability quality declines: lower is better for both ECE and Brier. A seeded candidate permutation changes 41/231 predictions and reduces accuracy by 7 decisions. Original-only correct: 17; shuffled-only correct: 10; correct in both: 119; incorrect in both: 85.

The local 4,096-token run covered 230/231 tasks. `hard-opus-a-long_policy-01` rendered to 4,118 tokens and failed without truncation. Among successful local requests, 35 prompts exceeded 2,048 tokens and eight exceeded 3,072.

## Accuracy by family

| Family | N | Base | Adapter | Shuffled | Jev |
|---|---:|---:|---:|---:|---:|
| adequacy | 12 | 6/12 | 6/12 | 6/12 | 12/12 |
| adversarial | 6 | 1/6 | 3/6 | 4/6 | 6/6 |
| ambiguous | 7 | 2/7 | 3/7 | 3/7 | 6/7 |
| extraction | 24 | 21/24 | 19/24 | 20/24 | 24/24 |
| fact | 12 | 7/12 | 11/12 | 10/12 | 12/12 |
| intent | 24 | 21/24 | 21/24 | 19/24 | 24/24 |
| judge_hard | 17 | 7/17 | 7/17 | 7/17 | 13/17 |
| long_policy | 19 | 4/19 | 7/19 | 4/19 | 12/19 |
| multi_hop | 18 | 3/18 | 4/18 | 4/18 | 16/18 |
| ordinal | 12 | 7/12 | 7/12 | 6/12 | 12/12 |
| policy | 12 | 6/12 | 6/12 | 6/12 | 11/12 |
| probability | 10 | 3/10 | 7/10 | 5/10 | 8/10 |
| routing | 12 | 6/12 | 8/12 | 8/12 | 12/12 |
| routing_hard | 5 | 1/5 | 4/5 | 5/5 | 5/5 |
| temporal_numeric | 15 | 5/15 | 7/15 | 6/15 | 3/15 |
| tool_selection | 12 | 12/12 | 12/12 | 12/12 | 12/12 |
| tradeoff | 6 | 5/6 | 4/6 | 4/6 | 5/6 |
| trap | 8 | 1/8 | 0/8 | 0/8 | 8/8 |

## Interpretation

The fine-tune learns useful decision behavior but remains much weaker than Jev on multi-step rules, traps, ambiguity, long policy interpretation, and answer-quality judging. It also has a strong affirmative bias: in binary Noul tasks it selected `yes` 67 times and `no` seven times. This explains much of the weakness on policy, adequacy, judging, and trap cases.

Candidate order is a material but secondary problem. The adapter still loses 65 correct decisions relative to Jev in its original order, so order averaging alone cannot close the gap. The next training run should balance true/false outcomes and candidate positions, include explicit none/unknown and trap cases, train multi-step rule application, and select checkpoints on both original and permuted validation orders.

Fit temperature calibration only on a disjoint validation set. The current adapter improves accuracy while worsening ECE and Brier, demonstrating that its raw candidate softmax should not be treated as calibrated. Keep all JevBench public tasks out of training and calibration if these results remain the reported evaluation.

## Reproduction

```sh
./.venv/bin/python jevbench_compare.py --providers both --tiers easy,standard,hard --max-tokens 4096 --out runs/jevbench-public-full-v2.json
./.venv/bin/python jevbench_compare.py --providers local --tiers easy,standard,hard --adapter none --max-tokens 4096 --out runs/jevbench-public-base-v1.json
./.venv/bin/python jevbench_compare.py --providers local --tiers easy,standard,hard --max-tokens 4096 --shuffle-seed 17 --out runs/jevbench-public-adapter-shuffled-v1.json
```

Each run checkpoints one JSONL record per task under `<output>.partial/` and resumes by task ID when the exact command is repeated. Credentials are never written to result files.
