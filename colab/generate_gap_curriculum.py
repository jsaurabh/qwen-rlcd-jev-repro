#!/usr/bin/env python3
"""Generate rule-grounded contrastive decision data for the Nimble/Jev reproduction."""
from __future__ import annotations

import argparse, hashlib, json, random
from pathlib import Path


def dump(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def row(rid, family, domain, state, instructions, criteria, target, split, meta):
    return {
        "id": rid, "family": family, "source_family": family, "domain": domain,
        "split": split,
        "input": {"state": state, "questions": {"decision": {
            "type": "choice", "instructions": instructions, "criteria": criteria}}},
        "reference": {"target": target, "source": "deterministic_rule_engine", "human_reviewed": False},
        "generation": meta,
    }


def temporal_pair(rng, i, split):
    limit = rng.choice([3, 5, 7, 10, 14, 30]); inside = rng.randint(0, limit)
    outside = limit + rng.randint(1, 5); amount = rng.choice([18, 24, 35, 50, 75])
    family = f"gap-temporal-{i:04d}"
    criteria = {
        "apply_adjustment": f"Apply the ${amount} adjustment when notice arrives no later than {limit} calendar days after posting.",
        "use_fallback": "Use the recorded fallback when the adjustment is ineligible and the customer supplied one.",
        "manual_review": "Use only if timing or authorization evidence is missing or contradictory.",
    }
    instructions = (f"Choose the authorized resolution. The adjustment window is inclusive: notice on day {limit} is timely. "
                    "A documented fallback controls only after the window closes.")
    def make(days, suffix, target):
        state = [
            {"speaker":"Policy record","text":f"The account has a valid ${amount} adjustment authorization for this posting."},
            {"speaker":"Customer instruction","text":"If the adjustment cannot be applied, close the disputed service and return any permitted unused balance."},
            {"speaker":"Timeline audit","text":f"The notice arrived {days} calendar days after the posting date; both timestamps are verified."},
            {"speaker":"Reconciliation","text":"There is one posting, no reversal, and no conflicting timestamp."},
        ]
        meta={"generator":"gap-curriculum-v1","rule":"inclusive_elapsed_days","limit":limit,"observed":days,"pair_role":suffix}
        return row(f"{family}-{suffix}",family,"banking",state,instructions,criteria,target,split,meta)
    return [make(inside,"base","apply_adjustment"), make(outside,"counterfactual","use_fallback")]


def numeric_pair(rng, i, split):
    threshold=rng.choice([500,1000,2500,5000,10000]); low=threshold-rng.choice([1,10,50]); high=threshold+rng.choice([1,10,50])
    family=f"gap-numeric-{i:04d}"
    criteria={
        "standard_review":f"Use when aggregate exposure is below ${threshold:,} and no override applies.",
        "enhanced_review":f"Use when aggregate exposure is at least ${threshold:,}, including equality.",
        "insufficient_data":"Use only when an included balance cannot be determined.",
    }
    instructions=("Choose the review tier. Aggregate exposure is the sum of cleared principal and pending principal; "
                  "exclude fees and reversed items. The enhanced threshold is inclusive.")
    def make(total,suffix,target):
        cleared=total-rng.randint(20,200); pending=total-cleared; fee=rng.choice([5,12,25])
        state=[
            {"speaker":"Ledger","text":f"Cleared principal is ${cleared:,}; pending principal is ${pending:,}."},
            {"speaker":"Fee ledger","text":f"A ${fee} processing fee is present and must be excluded from exposure."},
            {"speaker":"Operations","text":"All included balances are verified; there are no overrides or reversed principal items."},
        ]
        meta={"generator":"gap-curriculum-v1","rule":"inclusive_sum_threshold","threshold":threshold,"total":total,"pair_role":suffix}
        return row(f"{family}-{suffix}",family,"banking",state,instructions,criteria,target,split,meta)
    return [make(low,"base","standard_review"),make(high,"counterfactual","enhanced_review")]


def probability_pair(rng, i, split):
    threshold=rng.choice([0.6,0.7,0.75,0.8,0.9]); below=round(threshold-0.01,2); above=threshold
    family=f"gap-probability-{i:04d}"
    criteria={
        "auto_clear":f"Clear automatically only when calibrated legitimacy probability is at least {threshold:.0%} and no hard flag exists.",
        "analyst_review":"Send to an analyst when the probability is below the automatic threshold or a hard flag exists.",
        "decline":"Decline only when a verified mandatory-decline condition is present.",
    }
    instructions="Choose the disposition. Threshold comparisons are inclusive. A model score is a probability, not a percentage-point margin."
    def make(p,suffix,target):
        state=[
            {"speaker":"Risk model","text":f"The calibrated probability that the activity is legitimate is {p:.2f}."},
            {"speaker":"Controls","text":"No mandatory-decline condition or hard flag is present."},
            {"speaker":"Data quality","text":"The score is current, calibrated, and based on complete required fields."},
        ]
        meta={"generator":"gap-curriculum-v1","rule":"inclusive_probability_threshold","threshold":threshold,"observed":p,"pair_role":suffix}
        return row(f"{family}-{suffix}",family,"banking",state,instructions,criteria,target,split,meta)
    return [make(below,"base","analyst_review"),make(above,"counterfactual","auto_clear")]


def long_policy_pair(rng, i, split):
    threshold=rng.choice([2,3,4,5]); base=threshold-1; counter=threshold
    family=f"gap-policy-{i:04d}"
    criteria={
        "retain_standard_controls":"Retain standard controls when fewer than the required independent indicators are verified.",
        "apply_enhanced_controls":"Apply enhanced controls when the minimum independent-indicator count is met and all prerequisites hold.",
        "escalate_missing_evidence":"Escalate only when a required prerequisite cannot be established.",
    }
    instructions=(f"Choose the control set. Enhanced controls require at least {threshold} independent indicators, verified identity, and an open account. "
                  "Repeated observations from one source count once. Closed accounts follow archival policy. Missing evidence means unknown, not false.")
    distractors=[
        "The archival schedule is reviewed each January and does not change control eligibility.",
        "Customer preference affects notification channel but never the indicator count.",
        "A prior resolved alert is retained for audit and is not a current indicator.",
        "Service-tier status changes response time but not the eligibility rule.",
    ]
    def make(count,suffix,target):
        state=[{"speaker":"Case status","text":"Identity is verified and the account is open."}]
        for n in range(count):
            state.append({"speaker":f"Independent source {n+1}","text":f"Indicator {n+1} is current, verified, and independent of every other listed source."})
        state += [{"speaker":"Policy appendix","text":d} for d in rng.sample(distractors,len(distractors))]
        state.append({"speaker":"Audit","text":f"Exactly {count} qualifying independent indicators are present; duplicated observations were removed."})
        meta={"generator":"gap-curriculum-v1","rule":"conjunctive_count_threshold","threshold":threshold,"observed":count,"pair_role":suffix}
        return row(f"{family}-{suffix}",family,"banking",state,instructions,criteria,target,split,meta)
    return [make(base,"base","retain_standard_controls"),make(counter,"counterfactual","apply_enhanced_controls")]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--pairs-per-family",type=int,default=150); ap.add_argument("--seed",type=int,default=20260921)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True); rng=random.Random(a.seed)
    rows=[]; builders=[temporal_pair,numeric_pair,probability_pair,long_policy_pair]
    for b in builders:
        for i in range(a.pairs_per_family):
            split="eval" if i%5==0 else "train"; rows.extend(b(rng,i,split))
    rng.shuffle(rows); train=[r for r in rows if r["split"]=="train"]; ev=[r for r in rows if r["split"]=="eval"]
    assert not ({r['family'] for r in train}&{r['family'] for r in ev})
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:
        q=r['input']['questions']['decision']; assert r['reference']['target'] in q['criteria']
    dump(a.output/'all.jsonl',rows); dump(a.output/'train.jsonl',train); dump(a.output/'eval.jsonl',ev)
    manifest={"pipeline":"gap-curriculum-v1","seed":a.seed,"rows":len(rows),"train_rows":len(train),"eval_rows":len(ev),"pairs":len(rows)//2,"human_reviewed":False,"labels":"deterministic_rule_engine","families":sorted({r['source_family'].split('-')[1] for r in rows})}
    manifest['sha256']=hashlib.sha256((a.output/'all.jsonl').read_bytes()).hexdigest()
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
