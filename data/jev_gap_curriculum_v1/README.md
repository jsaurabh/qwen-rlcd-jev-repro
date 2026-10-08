# Jev gap curriculum v1

Public synthetic research dataset for the Qwen/Nimble reproduction. It targets the
weakest measured JevBench families: temporal/numeric comparisons, probability
thresholds, and long-policy conjunctions.

The dataset contains 1,200 contrastive pairs (2,400 rows). Each pair changes one
decision-bearing value and flips the target. Labels are calculated by Python
rules; no evaluated model or Jev output is used as a label. Pair families are
kept wholly in train or eval (1,920/480 rows) to prevent near-duplicate leakage.

Generate it on Colab or locally:

```bash
python generate_gap_curriculum.py \
  --output data/jev_gap_curriculum_v1 \
  --pairs-per-family 300 \
  --seed 20260921
```

This version is intentionally narrow. It is suitable for an ablation that tests
whether focused, rule-grounded contrastive data improves the identified weak
families. It is not a reproduction of Nimble's private/generated source pool and
has not been human reviewed. The dataset contains no real customer or banking
records and is released under the repository's MIT license.
