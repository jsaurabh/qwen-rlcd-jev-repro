# Data

Every dataset committed here is synthetic. No customer, transaction, or institution data is present.

## MLX frozen splits

`mlx/` contains the exact JSONL splits written by the corresponding M1 experiment before model evaluation. Each directory includes a manifest with row counts and SHA-256 hashes. The generators remain in the repository:

- `faithful-v1` → `banking_data.py`
- `compositional-v2` → `banking_v2_data.py`
- `banking-contrastive-v3` → `banking_contrastive_data.py`
- `banking-v4` → `banking_v4_data.py`

## Gap curriculum

`jev_gap_curriculum_v1/` contains 1,200 contrastive pairs across temporal, numeric, probability-threshold, and long-policy rules. Labels come from deterministic Python rules. Whole pair families are assigned to train or evaluation; no pair is split across them.

Regenerate it with:

```bash
python3 tools/generate_gap_curriculum.py \
  --output data/jev_gap_curriculum_v1 \
  --pairs-per-family 300 \
  --seed 20260921
```

These datasets are suitable for code-path reproduction and targeted ablations. They have not been reviewed by banking professionals and cannot establish production quality.
