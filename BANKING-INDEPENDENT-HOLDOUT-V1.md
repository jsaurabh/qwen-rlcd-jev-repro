# Independent banking holdout v1

The v3 adapter’s perfect in-template synthetic test score does not establish reliable performance on this independently structured holdout. The following evaluation was frozen before inference and did not retrain, select a checkpoint, tune prompts against results, or choose probability thresholds.

## Design and provenance

48 entirely synthetic cases: eight policy families × two writing structures × three outcomes. Each of yes, no, and insufficient_evidence has 16 cases. Eight unknown cases omit a decisive case fact; eight omit the governing policy. No cases or labels received independent human review or banking-professional review. Policies are invented test rules, not claims about banking law or real institutions.

The new rules cover dual consent, an overriding restraint, elapsed-time boundaries, alternative proof, tiered coverage, policy version precedence, document scope, and aggregate allowance. The products and channels are listed in family_inventory.json. All are absent from the v3 generator and saved training records. Dialogue handovers and nested document inventories replace the training policy/case_fact/request template. Exact policy/product/channel strings, source IDs, and state overlap were checked. Rule-structure and wording-family separation is an author audit; this is not proof of absence from base-model pretraining, and generic banking concepts can overlap.

Only the selected step-64 adapter in runs/banking-contrastive-v3/best and its quantized base were evaluated. The pinned reference-nimble commit is f136b3f75721fda4ea961f73993cc50b08488835. Its existing as_scoring/encode_scoring and candidate-logit prompt path are reused without modification. The original prompt SHA-256 and training-data SHA-256 match v3. All six permutations of the three candidates were specified before inference, with argmax decisions and a 768-token cap; no truncation occurred. All judgments use three-way Choice because binary Noul cannot express unknown. Thus semantic and output-schema generalization are confounded; this is not a like-for-like repeat of v3’s binary tests.

## Results

V3 accuracy ranges from 16/48 to 17/48 (33.3–35.4%); base ranges from 16/48 to 23/48 (33.3–47.9%). V3 predicts yes on 46–48 cases per order and never predicts no. Its semantic answer stays unchanged across all six orders on 46/48 cases, versus 16/48 for the base; this stability largely reflects a yes bias. V3 is correct across every order on 16/48 cases versus 6/48 for the base, so the models trade off per-order accuracy and robustness. V3 recognizes no missing-policy cases and only one missing-fact case in one order. These results do not support dependable transfer or calibrated confidence on this holdout.

| Model | Candidate order | Correct | Unknown correct | NLL | Brier | ECE (10 bins) |
|---|---|---:|---:|---:|---:|---:|
| base | 0: yes / no / insufficient_evidence | 17/48 | 0/16 | 2.871 | 1.107 | 0.494 |
| base | 1: yes / insufficient_evidence / no | 21/48 | 6/16 | 2.046 | 0.899 | 0.382 |
| base | 2: no / yes / insufficient_evidence | 16/48 | 0/16 | 4.368 | 1.300 | 0.653 |
| base | 3: no / insufficient_evidence / yes | 23/48 | 11/16 | 1.307 | 0.722 | 0.264 |
| base | 4: insufficient_evidence / yes / no | 16/48 | 0/16 | 3.885 | 1.309 | 0.656 |
| base | 5: insufficient_evidence / no / yes | 17/48 | 0/16 | 2.678 | 0.934 | 0.388 |
| v3 | 0: yes / no / insufficient_evidence | 16/48 | 0/16 | 10.309 | 1.323 | 0.658 |
| v3 | 1: yes / insufficient_evidence / no | 17/48 | 1/16 | 9.693 | 1.240 | 0.622 |
| v3 | 2: no / yes / insufficient_evidence | 16/48 | 0/16 | 7.248 | 1.333 | 0.667 |
| v3 | 3: no / insufficient_evidence / yes | 16/48 | 0/16 | 5.992 | 1.333 | 0.666 |
| v3 | 4: insufficient_evidence / yes / no | 16/48 | 0/16 | 5.776 | 1.332 | 0.666 |
| v3 | 5: insufficient_evidence / no / yes | 16/48 | 0/16 | 6.730 | 1.333 | 0.667 |

| Model | Correct under every order | Same semantic answer under every order |
|---|---:|---:|
| base | 6/48 | 16/48 |
| v3 | 16/48 | 46/48 |

Missing evidence results (correct of eight, orders 0–5):

```json
{
  "base": {
    "policy_missing_correct_by_order": [
      0,
      4,
      0,
      8,
      0,
      0
    ],
    "case_fact_missing_correct_by_order": [
      0,
      2,
      0,
      3,
      0,
      0
    ]
  },
  "v3": {
    "policy_missing_correct_by_order": [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    "case_fact_missing_correct_by_order": [
      0,
      1,
      0,
      0,
      0,
      0
    ]
  }
}
```

Candidate permutations reuse the same 48 cases; they are not 288 independent observations. The two writing variants and three outcomes are correlated within eight policy families. Fixed-bin ECE is descriptive on this small synthetic set and does not establish calibration. Existing evaluator “contrastive_family_accuracy” counts complete three-case groups here (16 groups), not eight policy families; pair accuracy is null because there are no two-record pairs. No deployment gate or threshold was selected from this holdout. Treat these cases as consumed evaluation data for future work.

## Exact commands

Run from the repository root:

```sh
.venv/bin/python banking_independent_holdout.py --out runs/banking-independent-holdout-repeat
```

The evaluator creates runs/banking-independent-holdout-v1 and refuses to reuse an existing output directory. For an explicit repeat, pass a new --out directory. contract.json is written before any model inference and records dataset, evaluator, prompt, model, adapter, and prior-run hashes. All original v3 run and base-model files were hash-checked again after inference.

Prompt isolation was verified for every label on all 48 cases (144 checks): changing gold labels, rationale, and provenance did not change inference tokens. Repeat it with:

```sh
.venv/bin/python tools/verify_prompt_isolation.py --run runs/banking-independent-holdout-repeat
```

Verification: 144 prompt-isolation checks passed: labels and provenance do not change inference tokens; 48 unique aligned cases in every evaluation; Every one of six permutations appears exactly once for each model; Semantic targets, argmax predictions and correctness agree with candidate positions; Probabilities finite and normalized; NLL and accuracy independently recomputed; All preexisting v3 run files and base model files unchanged (SHA-256 checked by evaluator).

## Artifacts and limitations

banking_independent_holdout.py contains the author-assigned test cases and evaluator. The new run contains holdout.jsonl, family_inventory.json, contract.json, twelve per-order probability files, report.json, verification.json, and evaluation.log. BANKING-INDEPENDENT-HOLDOUT-V1.md contains this report. Existing experiment runs were preserved.

The installed TypeSafe skill guided explicit insufficient-evidence outcomes and separation of uncertainty from correctness. Attempts to read https://docs.typesafe.ai/llms.txt and https://docs.typesafe.ai/primitives/choice.md failed in the web tool; curl also failed DNS resolution. The installed skill and pinned local implementation were used as the fallback; no TypeSafe API calls or Jev benchmark were made in this stage.

Evaluation wall time: 175.6 seconds. Dataset SHA-256: `cd240be6c47271bf874b5c356a8b0e5e3ac53e6a57c4ee76904c214d59f14774`.
