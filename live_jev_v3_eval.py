"""Small labeled live comparison: banking v3 adapter versus TypeSafe Jev."""
import argparse, json, math, os, sys, time
from pathlib import Path
import mlx.core as mx
import httpx
from mlx_lm import load

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference-nimble'))
from nimble.training.schema_data import as_scoring, encode_scoring
from nimble_1_5b_mlx import candidate_logits, dump

CASES=[
 {"id":"live-suspected-fraud","state":{"policy":"Treat activity as suspected fraud when the customer reports it is unrecognized.","case_fact":"The customer reports that the new-device card purchase is unrecognized.","request":"Apply the policy."},"question":{"type":"noul","instructions":"Do the facts indicate suspected fraud?","criteria":{"false":"The event is recognized and expected.","true":"The event is unrecognized or suspicious."}},"target":True},
 {"id":"live-authorization","state":{"policy":"Authorization is disputed when the customer denies authorizing the transaction.","case_fact":"The customer confirms they authorized the ACH payment to the landlord.","request":"Apply the policy."},"question":{"type":"noul","instructions":"Does the customer dispute authorizing the transaction?","criteria":{"false":"The customer affirms authorization.","true":"The customer denies authorization."}},"target":False},
 {"id":"live-conflict","state":{"policy":"Information conflicts when the bank record and customer document state different material values.","case_fact":"The bank record shows $340, while the signed receipt shows $304.","request":"Apply the policy."},"question":{"type":"noul","instructions":"Do material records conflict?","criteria":{"false":"The material values agree.","true":"The material values disagree."}},"target":True},
 {"id":"live-exception","state":{"policy":"A policy exception is required when the customer asks the bank to bypass a stated rule.","case_fact":"The customer accepts the beneficiary hold and asks when it will expire.","request":"Apply the policy."},"question":{"type":"noul","instructions":"Does resolving this request require a policy exception?","criteria":{"false":"The request complies with the stated rule.","true":"The customer asks to bypass the stated rule."}},"target":False},
 {"id":"live-ongoing-loss","state":{"policy":"Ongoing loss exists when an unauthorized transfer remains active or pending.","case_fact":"The unauthorized wire has not been stopped and another debit is pending.","request":"Apply the policy."},"question":{"type":"noul","instructions":"Are unauthorized funds currently leaving or about to leave?","criteria":{"false":"No unauthorized transfer remains active or pending.","true":"An unauthorized transfer remains active or pending."}},"target":True},
 {"id":"live-blocked","state":{"policy":"The customer is blocked when they cannot access the account or complete the intended action.","case_fact":"The customer can sign in and can download the requested statement.","request":"Apply the policy."},"question":{"type":"noul","instructions":"Is the customer currently blocked?","criteria":{"false":"The customer can access and complete the action.","true":"The customer cannot access or complete the action."}},"target":False},
 {"id":"live-department","state":{"policy":"Route recognized transaction servicing to payments; route a transaction the customer denies authorizing to fraud review.","case_fact":"The customer denies authorizing the mobile transfer and asks for an investigation.","request":"Apply the routing policy."},"question":{"type":"choice","instructions":"Which specialist team should own this case?","criteria":{"payments":"A recognized transaction-service request.","fraud_review":"A transaction the customer denies authorizing."}},"target":"fraud_review"},
 {"id":"live-deadline","state":{"policy":"Use Routine for no deadline within a week, Prompt for a deadline within one business day, and Immediate for a deadline within hours.","case_fact":"The stated deadline is tomorrow afternoon.","request":"Apply the deadline rubric."},"question":{"type":"score","instructions":"Ignoring fraud, blockage, and ongoing loss, rate the independent deadline pressure.","criteria":["Routine: no deadline within the next week.","Prompt: a deadline within one business day.","Immediate: a deadline within hours."]},"target":1},
]

def key():
    value=os.environ.get('TYPESAFE_API_KEY','').strip();path=ROOT/'.env'
    if not value and path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('TYPESAFE_API_KEY='):value=line.split('=',1)[1].strip().strip('"').strip("'")
    if not value:raise SystemExit('TYPESAFE_API_KEY is missing')
    return value

def raw_record(case):
    return {'id':case['id'],'family':case['id'],'source_family':case['id'],'domain':'banking','split':'test','variant':'live',
            'input':{'state':case['state'],'questions':{'decision':case['question']}},'reference':{'target':case['target']}}

def local_score(model,tok,case):
    scoring=as_scoring(raw_record(case),False);encoded=encode_scoring(scoring,tok,768,None);row={'input_ids':encoded['prompt_token_ids'],'candidate_ids':encoded['candidate_token_ids'],'label':encoded['target_index']}
    started=time.perf_counter();z=candidate_logits(model,row);mx.eval(z);p=mx.softmax(z).tolist();elapsed=(time.perf_counter()-started)*1000
    choices=list(encoded['code_to_choice'].values());probs={str(v).lower() if isinstance(v,bool) else str(v):float(x) for v,x in zip(choices,p)}
    return {'prediction':choices[max(range(len(p)),key=p.__getitem__)],'probabilities':probs,'elapsed_ms':elapsed}

def jev_score(client,case,model_name):
    started=time.perf_counter();response=client.post('/v1/systemone',json={'state':case['state'],'questions':{'decision':case['question']},'model':model_name});elapsed=(time.perf_counter()-started)*1000
    if response.status_code!=200:raise RuntimeError(f'Jev HTTP {response.status_code}')
    body=response.json();answer=body['answers']['decision'];kind=case['question']['type']
    if kind=='noul':probs={'false':1-answer['noul'],'true':answer['noul']};prediction=answer['noul']>=.5
    elif kind=='choice':probs={str(k):float(v) for k,v in answer['probabilities'].items()};prediction=answer['choice']
    else:probs={str(k):float(v) for k,v in answer['probabilities'].items()};prediction=max(probs,key=probs.get)
    return {'prediction':prediction,'probabilities':probs,'elapsed_ms':elapsed,'model':body.get('model')}

def metric(rows,provider):
    correct=0;nll=0.;brier=0.
    for row in rows:
        target=str(row['target']).lower() if isinstance(row['target'],bool) else str(row['target']);p=row[provider]['probabilities'];pred=str(row[provider]['prediction']).lower() if isinstance(row[provider]['prediction'],bool) else str(row[provider]['prediction']);correct+=pred==target;nll-=math.log(max(p[target],1e-15));brier+=sum((v-int(k==target))**2 for k,v in p.items())
    return {'cases':len(rows),'correct':correct,'accuracy':correct/len(rows),'nll':nll/len(rows),'brier':brier/len(rows),'median_ms':sorted(r[provider]['elapsed_ms'] for r in rows)[len(rows)//2]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',default='runs/banking-contrastive-v3/best');ap.add_argument('--model',default='models/qwen-1.5b-4bit');ap.add_argument('--jev-model',default='jev-latest');ap.add_argument('--out',default='runs/live-jev-v3.json');args=ap.parse_args()
    out=ROOT/args.out
    if out.exists():ap.error('output exists; choose a new --out')
    model,tok=load(str(ROOT/args.model),adapter_path=str(ROOT/args.adapter));model.eval();rows=[]
    with httpx.Client(base_url='https://api.typesafe.ai',headers={'Authorization':'Bearer '+key()},timeout=60,follow_redirects=False) as client:
        for case in CASES:
            print('Scoring '+case['id'],flush=True);rows.append({'id':case['id'],'state':case['state'],'question':case['question'],'target':case['target'],'local':local_score(model,tok,case),'jev':jev_score(client,case,args.jev_model)})
    result={'protocol':{'cases':'fresh hand-authored deterministic policy cases','labels_visible_to_models':False,'same_state_and_question':True,'adapter':args.adapter,'jev_model_requested':args.jev_model,'api_key_saved':False},'summary':{'local':metric(rows,'local'),'jev':metric(rows,'jev'),'answer_agreement':sum(str(r['local']['prediction']).lower()==str(r['jev']['prediction']).lower() for r in rows)/len(rows)},'rows':rows}
    dump(out,result);print(json.dumps({'output':str(out),'summary':result['summary']},indent=2))

if __name__=='__main__':main()
