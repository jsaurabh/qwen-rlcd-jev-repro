# Data

The only committed dataset is `jev_gap_curriculum_v1/`: 1,200 synthetic contrastive pairs covering temporal, numeric, probability-threshold, and long-policy decisions. Labels come from deterministic Python rules, and entire pair families are assigned to train or evaluation so paired examples cannot cross the split.

Regenerate it with:

```bash
python3 tools/generate_gap_curriculum.py \
  --output data/jev_gap_curriculum_v1 \
  --pairs-per-family 300 \
  --seed 20260921
```

No customer, transaction, or institution data is included. This curriculum is suitable for code-path reproduction and targeted ablations; it has not been reviewed for production banking use.
