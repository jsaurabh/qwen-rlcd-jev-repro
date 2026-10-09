# Targeted data experiment (in progress)

The CPU overlap audit runs before any new GPU training. `audit_overlap.py` expects archived train/dev/temperature splits in `/content/data-v3/work/original` and private benchmark `*rows.jsonl.gz` in `/content/data-v3/work/suite-0.3`. Install `datasketch`, then run the script. Outputs are private cleaned JSONL and an aggregate report under `work/audit`.

The screen compares normalized states using exact SHA256 and MinHash candidates with exact shingle Jaccard confirmation. Near matching uses at most4096words,64permutations and threshold0.8. It can miss paraphrases and is not proof of contamination absence. It intentionally removes shared documents conservatively, even when question wording differs.

`experiment.json` is the proposed allocation and evaluation contract, not a claim that100k examples have been built or training has completed. Source licenses/access, revisions, candidate transformations and held-out family splits must pass before training. Benchmark prompts and source corpora stay private; publish only recipes, manifests and aggregate audit/results.
