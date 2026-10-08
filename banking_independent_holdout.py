"""Frozen synthetic family holdout; evaluation only, no optimization or thresholds."""
import argparse, copy, hashlib, itertools, json, subprocess, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference-nimble'))
from nimble.training.schema_data import as_scoring, encode_scoring
from nimble_1_5b_mlx import evaluate, sha, dump
from mlx_lm import load
import mlx.core as mx

# Authored independently of v3 templates. These are invented bank rules, not law.
SPECS=[
('dual_consent','custodial term certificate','secure document exchange','May the maturity instruction proceed?',
 'A maturity instruction may proceed only if both the custodian and the beneficiary have signed. One signature alone is not enough.',
 ['The custodian signed on Monday. The beneficiary signed on Tuesday.','The custodian signed on Monday. The beneficiary explicitly declined to sign.','The custodian signed on Monday. The beneficiary signature page is absent from this packet.'],
 'Both signatures are required; a missing page does not prove refusal.'),
('priority_override','estate distribution bond','probate desk','May the distribution be released?',
 'Release is permitted after probate approval, except that a court restraint overrides approval and prohibits release.',
 ['Probate approval is recorded. The court explicitly confirms there is no restraint.','Probate approval is recorded. A court restraint is in force.','Probate approval is recorded. The court restraint register could not be obtained.'],
 'The restraint overrides approval; unavailable restraint status leaves the result unresolved.'),
('elapsed_window','notice deposit','postal instruction centre','Is the notice period satisfied?',
 'The notice period is satisfied at 30 elapsed calendar days or more. The delivery date, not the date written, starts the clock.',
 ['The letter was written 35 days ago and delivered 30 days ago.','The letter was written 35 days ago and delivered 29 days ago.','The letter was written 35 days ago. Delivery tracking has been lost.'],
 'Use elapsed days since delivery, including the boundary at 30.'),
('alternative_proof','education escrow','campus liaison desk','Does the packet meet the enrollment proof rule?',
 'Enrollment proof is sufficient if either a registrar seal or a verified institution attestation is present. Neither is mandatory when the other is present.',
 ['No registrar seal is present. A verified institution attestation is enclosed.','No registrar seal is present. The institution confirms no attestation was issued.','No registrar seal is present. Staff have not checked whether an attestation is enclosed.'],
 'Either proof suffices; lack of a seal alone is inconclusive.'),
('tiered_security','warehouse receipt finance','trade documentation portal','Does the pledged coverage meet the requirement?',
 'Standard inventory requires coverage of at least 100 percent. Perishable inventory requires at least 120 percent.',
 ['The inventory is perishable. Pledged coverage is 120 percent.','The inventory is perishable. Pledged coverage is 119 percent.','Pledged coverage is 110 percent. The inventory classification was not supplied.'],
 'Apply the product class before comparing coverage; unknown class straddles the requirement.'),
('version_precedence','equipment lease reserve','dealer service kiosk','Is reserve reduction allowed?',
 'For contracts signed before June 1, reduction is allowed after six installments. For contracts signed on or after June 1, eight installments are required.',
 ['The contract was signed June 1. Eight installments are recorded.','The contract was signed June 1. Seven installments are recorded.','Seven installments are recorded. The contract signature date is unreadable.'],
 'The signature date determines which installment rule applies.'),
('scope_exclusion','documentary collection','correspondent message hub','Is translation required for this packet?',
 'Translation is required for a foreign-language original. A packet containing only a domestic-language original and a foreign-language courtesy copy is exempt.',
 ['The sole original is in a foreign language. No domestic-language original exists.','The original is in the domestic language. Only the courtesy copy is in a foreign language.','A foreign-language document is present. Its status as original or courtesy copy is missing.'],
 'Document role matters, not merely the presence of a foreign language.'),
('aggregate_cap','charitable endowment draw','trust committee hearing','Is this draw within the annual allowance?',
 'The annual allowance is 50 units. Add the proposed draw to draws already paid this year. A combined total of at most 50 is within allowance.',
 ['Draws already paid total 35 units. The proposed draw is 15 units.','Draws already paid total 35 units. The proposed draw is 16 units.','The proposed draw is 15 units. The year-to-date payout ledger is unavailable.'],
 'Compare the combined total with the inclusive cap; missing history prevents that calculation.')]
CRITERIA={'yes':'The supplied rule and evidence establish that the stated condition is met.',
 'no':'The supplied rule and evidence establish that the stated condition is not met.',
 'insufficient_evidence':'A necessary rule or case fact is missing or unresolved, so neither yes nor no is established.'}

def build():
 rows=[]
 for family,product,channel,q,policy,facts,rationale in SPECS:
  for style in ('handover_dialogue','document_inventory'):
   for j,label in enumerate(CRITERIA):
    missing_policy=style=='document_inventory' and j==2
    evidence=facts[0] if missing_policy else facts[j]
    rule='Rule attachment is absent; no applicable rule is supplied.' if missing_policy else policy
    if style=='handover_dialogue':
     state=f'Intake officer: This concerns {product}, received through {channel}.\nReviewer: What governs it?\nIntake officer: {rule}\nReviewer: What do we actually have?\nIntake officer: {evidence}\nReviewer: Treat omissions as unresolved, not as negative facts.'
    else:
     state={'dossier':{'instrument':product,'received_via':channel},'attachments':[{'title':'governing provision','contents':rule},{'title':'case evidence','contents':evidence}],'recordkeeping_note':'Missing attachments convey no affirmative or negative finding.'}
    group=f'independent-{family}-{style}'
    rows.append({'id':f'{group}-{label}','family':group,'source_family':family,'domain':'banking','split':'holdout',
      'input':{'state':state,'questions':{'decision':{'type':'choice','instructions':q+' Apply only the supplied fictional bank rule. Use insufficient_evidence if a necessary rule or fact is absent.','criteria':CRITERIA}}},
      'reference':{'target':label,'source':'synthetic_author_assigned','human_reviewed':False,'rationale':'Applicable rule absent.' if missing_policy else rationale},
      'provenance':{'synthetic':True,'human_reviewed':False,'policy_family':family,'product_family':product,'channel_family':channel,'wording_family':style,'missing_evidence':'policy' if missing_policy else 'case_fact' if j==2 else None}})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs/banking-independent-holdout-v1');args=ap.parse_args();out=ROOT/args.out
 out.mkdir(parents=True,exist_ok=False)
 rows=build();train=ROOT/'runs/banking-contrastive-v3/data/train.jsonl';training=[json.loads(x) for x in train.read_text().splitlines()]
 training_text=train.read_text().lower()
 assert len(rows)==48 and Counter(r['reference']['target'] for r in rows)=={'yes':16,'no':16,'insufficient_evidence':16}
 assert all(s[1].lower() not in training_text and s[2].lower() not in training_text and s[4].lower() not in training_text for s in SPECS)
 assert not {r['source_family'] for r in rows}&{r['source_family'] for r in training}
 assert not {json.dumps(r['input']['state'],sort_keys=True) for r in rows}&{json.dumps(r['input']['state'],sort_keys=True) for r in training}
 pin=subprocess.check_output(['git','-C',str(ROOT/'reference-nimble'),'rev-parse','HEAD'],text=True).strip()
 assert pin=='f136b3f75721fda4ea961f73993cc50b08488835'
 v3=json.loads((ROOT/'runs/banking-contrastive-v3/report.json').read_text());assert v3['selected_step']==64 and v3['accepted']
 assert sha(train)==v3['contract']['dataset']['train']['sha256']
 prompt=ROOT/'reference-nimble/nimble/scoring/parallel_schema.py';assert sha(prompt)==v3['contract']['prompt_code_sha256']
 data=out/'holdout.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 protected=[p for p in (ROOT/'runs/banking-contrastive-v3').rglob('*') if p.is_file()]+list((ROOT/'models/qwen-1.5b-4bit').glob('*'))
 hashes={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
 orders=list(itertools.permutations(CRITERIA))
 contract={'created_before_inference':True,'data_sha256':sha(data),'code_sha256':sha(__file__),'nimble_commit':pin,'prompt_sha256':sha(prompt),'protected_sha256':hashes,
 'rows':48,'policy_families':8,'wording_families':2,'orders':orders,'decision':'candidate argmax; no calibration, threshold selection, checkpoint selection or training',
 'adapter':'runs/banking-contrastive-v3/best','selected_step':64,'max_tokens':768,'evidence':'Entirely synthetic, author-assigned labels; no independent human or banking-professional review.',
 'separation':'All eight rule structures, eight product families, eight channel families and two document structures are newly authored and excluded from the v3 training generator. Exact product/channel/policy and state/source overlap checks passed. Semantic separation is an author audit, not proof of pretraining exclusion.',
 'schema_shift':'All judgments use three-way Choice to represent unknown; v3 mainly trained binary Noul. Results jointly stress semantic and output-schema generalization.',
 'metrics':'Six exhaustive orders; semantic agreement and correctness across all orders; unknown recall; known-to-unknown rate; NLL, Brier and fixed ten-bin ECE. Orders and correlated variants are not independent samples.'}
 dump(out/'contract.json',contract);dump(out/'family_inventory.json',[{'policy_family':s[0],'product':s[1],'channel':s[2],'question':s[3],'policy':s[4]} for s in SPECS])
 mx.set_cache_limit(1024**3);result={};started=time.perf_counter()
 for name,adapter in [('base',None),('v3',str(ROOT/'runs/banking-contrastive-v3/best'))]:
  model,tok=load(str(ROOT/'models/qwen-1.5b-4bit'),adapter_path=adapter);runs=[]
  for k,order in enumerate(orders):
   encoded=[]
   for raw in rows:
    scoring=as_scoring(raw,False);scoring['schema']['decision']['choices']=list(order)
    item=encode_scoring(scoring,tok,768)
    encoded.append({'id':raw['id'],'pair':raw['family'],'family':raw['source_family'],'domain':'banking','kind':'choice','input_ids':item['prompt_token_ids'],'candidate_ids':item['candidate_token_ids'],'label':item['target_index'],'choices':list(item['code_to_choice'].values())})
   records,summary,seconds=evaluate(model,encoded)
   for r,raw in zip(records,rows):
    r['semantic_prediction']=r['choices'][r['prediction']];r['target']=raw['reference']['target'];r['provenance']=raw['provenance']
   unknown=[r for r in records if r['target']=='insufficient_evidence'];known=[r for r in records if r['target']!='insufficient_evidence']
   summary.update(unknown_recall=sum(r['correct'] for r in unknown)/len(unknown),known_to_unknown_rate=sum(r['semantic_prediction']=='insufficient_evidence' for r in known)/len(known),by_policy={f:sum(r['correct'] for r in records if r['family']==f)/6 for f in {r['family'] for r in records}},max_prompt_tokens=max(len(r['input_ids']) for r in encoded))
   runs.append(records);dump(out/f'{name}-order-{k}.json',{'order':order,'summary':summary,'seconds':seconds,'rows':records});print(name,k,json.dumps(summary),flush=True)
  result[name]={'orders':[json.loads((out/f'{name}-order-{k}.json').read_text())['summary'] for k in range(6)],'all_order_accuracy':sum(all(run[i]['correct'] for run in runs) for i in range(48))/48,'semantic_agreement':sum(len({run[i]['semantic_prediction'] for run in runs})==1 for i in range(48))/48}
  del model;mx.clear_cache()
 assert all(sha(ROOT/p)==h for p,h in hashes.items())
 dump(out/'report.json',{'contract':contract,'models':result,'wall_seconds':time.perf_counter()-started,'protected_files_unchanged':True})
if __name__=='__main__':main()
