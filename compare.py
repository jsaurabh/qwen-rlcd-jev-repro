"""Compare unchanged community Qwen-RLCD, our scorers and official Jev API."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import httpx
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from mlx_lm import load
from mlx_jev_repro import SCHEMA, DecisionEngine
from core import engine_mlx
from core.schema import StructuredSchema

# Fresh, authored policy examples. Labels are rubric judgments, not real bank outcomes.
CASES=[
 ('A customer asks for the status of a scheduled transfer. It is on time and no purchase is blocked.',[0,0,0]),
 ('A customer asks for an explanation of a card charge they recognize. Nothing is disputed.',[0,0,0]),
 ('A recognized transfer is overdue and preventing a purchase. There is no fraud or policy exception.',[0,1,0]),
 ('A customer asks to bypass a transfer restriction. Their payment is blocked pending this policy exception.',[0,1,1]),
 ('A customer asks how to change their postal address. They can sign in normally.',[1,0,0]),
 ('A customer cannot sign in after forgetting their password. They report no suspicious activity.',[1,1,0]),
 ('Access restoration is blocked because the supplied ownership records contradict each other.',[1,1,1]),
 ('A customer wants to know where to download account statements. Account access is working.',[1,0,0]),
 ('A customer disputes authorization of a completed charge. The card is frozen and no further activity is occurring.',[2,1,1]),
 ('Unknown transfers are leaving an account right now. Another unauthorized transfer is pending.',[2,2,1]),
 ('A thief is using stolen account credentials to withdraw funds at this moment.',[2,2,1]),
 ('A customer reports an unauthorized payment scheduled to leave in five minutes. It has not been stopped.',[2,2,1]),
]
RUBRIC=['routine inquiry with no immediate disruption','access or payment disruption needing prompt attention','suspected ongoing unauthorized activity or imminent loss']


def write(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def valid_ps(ps):
    result=[]
    if len(ps)!=len(SCHEMA): raise ValueError('Missing fields')
    for p,s in zip(ps,SCHEMA):
        p=np.array(p,dtype=float)
        if len(p)!=len(s['choices']) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or abs(p.sum()-1)>.002:
            raise ValueError('Invalid candidate distribution')
        result.append((p/p.sum()).tolist()) # only small wire-rounding correction
    return result


def api_questions():
    questions={}
    for s in SCHEMA:
        q={'type':s['type'],'instructions':s['question']}
        if s['type']=='choice':q['criteria']={c:None for c in s['choices']}
        if s['type']=='score':q['criteria']=RUBRIC
        questions[s['name']]=q
    return questions


def read_key():
    key=os.environ.get('TYPESAFE_API_KEY','').strip()
    if not key and (ROOT/'.env').exists():
        for line in (ROOT/'.env').read_text().splitlines():
            if line.startswith('TYPESAFE_API_KEY='):
                key=line.split('=',1)[1].strip().strip('"').strip("'")
    if not key: raise SystemExit('Set TYPESAFE_API_KEY in the project .env file, then rerun --backend jev.')
    return key


def summarize(records):
    successful=[r for r in records if 'probabilities' in r]
    result={'cases':len(records),'successful':len(successful),'errors':len(records)-len(successful)}
    if not successful:return result
    predictions=[];labels=[];conf=[];brier=[];nll=[];perfield=[[],[],[]]
    for r in successful:
        for i,(p,y) in enumerate(zip(r['probabilities'],r['labels'])):
            p=np.array(p);pred=int(p.argmax())
            predictions.append(pred);labels.append(y);conf.append(float(p.max()))
            perfield[i].append(float(pred==y))
            nll.append(-math.log(max(float(p[y]),1e-12)))
            brier.append(float(np.sum((p-np.eye(len(p))[y])**2)))
    correct=np.array(predictions)==labels;conf=np.array(conf)
    ece=0.
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any():ece+=float(mask.mean()*abs(correct[mask].mean()-conf[mask].mean()))
    result.update(accuracy=float(correct.mean()),nll=float(np.mean(nll)),brier=float(np.mean(brier)),ece_5_bins=ece,
        field_accuracy={s['name']:float(np.mean(x)) for s,x in zip(SCHEMA,perfield)},
        median_ms=float(np.median([r['elapsed_ms'] for r in successful])),p95_ms=float(np.percentile([r['elapsed_ms'] for r in successful],95)))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['upstream','ours_base','ours_adapter','jev','local'],default='local')
    ap.add_argument('--model',default='models/qwen-1.5b-4bit')
    ap.add_argument('--adapter',default='runs/banking-1.5b')
    ap.add_argument('--jev-model',default='jev-latest')
    ap.add_argument('--out',default='runs/comparison-banking')
    args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':SCHEMA,'cases':[{'id':i,'state':x,'labels':y} for i,(x,y) in enumerate(CASES)],'score_rubric':RUBRIC}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    write(out/'cases.json',manifest)
    backends=['upstream','ours_base','ours_adapter'] if args.backend=='local' else [args.backend]
    for backend in backends:
        client=None
        metadata={'model':args.jev_model if backend=='jev' else args.model,'adapter':args.adapter if backend=='ours_adapter' else None,'temperature':1.,'dataset_sha256':digest,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if backend=='jev':
            client=httpx.Client(base_url='https://api.typesafe.ai',headers={'Authorization':'Bearer '+read_key()},timeout=30.,follow_redirects=False)
            def run(state):
                response=client.post('/v1/systemone',json={'state':state,'model':args.jev_model,'questions':api_questions()})
                if response.status_code!=200:raise RuntimeError('Jev HTTP '+str(response.status_code))
                raw=response.json();answers=raw['answers']
                ps=[]
                for s in SCHEMA:
                    a=answers[s['name']]
                    if a['type']!=s['type']:raise ValueError('Unexpected Jev answer type')
                    if s['type']=='noul':ps.append([1-a['noul'],a['noul']])
                    else:ps.append([a['probabilities'][str(c)] for c in s['choices']])
                return valid_ps(ps),raw
        else:
            model,tok=load(str(ROOT/args.model),adapter_path=str(ROOT/args.adapter) if backend=='ours_adapter' else None)
            model.eval()
            if backend=='upstream':
                engine_mlx._model,engine_mlx._tokenizer=model,tok
                schema=StructuredSchema({s['name']:{'type':'boolean' if s['type']=='noul' else 'enum','description':s['question'],'choices':None if s['type']=='noul' else [str(c) for c in s['choices']]} for s in SCHEMA})
                meta=schema.compile_parallel_metadata(tok)
                if any(meta['has_collisions']):raise ValueError('Upstream token collision: its fallback confidence is unsuitable for calibration comparison')
                def run(state):
                    raw=engine_mlx.run_parallel_generation(state,schema,temperature=1.)
                    ps=[]
                    for s in SCHEMA:
                        lookup={c['choice']:c['probability'] for c in raw['field_telemetry'][s['name']]['top_choices']}
                        ps.append([lookup[str(c).lower() if isinstance(c,bool) else str(c)] for c in s['choices']])
                    return valid_ps(ps),raw
            else:
                engine=DecisionEngine(model,tok,384)
                if backend=='ours_adapter':
                    metadata['temperature']=json.loads((ROOT/args.adapter/'calibration.json').read_text())['temperature']
                def run(state):
                    raw=engine.decide(state,metadata['temperature'])
                    return valid_ps([[c['probability'] for c in raw[s['name']]['distribution']] for s in SCHEMA]),raw
            # Exclude model load and shader warmup; fresh state cache for every timed case.
            run('A customer asks for a routine statement copy.')
        records=[]
        for i,(state,labels) in enumerate(CASES):
            t=time.perf_counter();r={'id':i,'labels':labels}
            try:
                ps,raw=run(state);r.update(probabilities=ps,raw=raw)
            except Exception as e:
                # Never serialize request headers, credentials or provider error bodies.
                r['error']=type(e).__name__
                if backend=='jev' and isinstance(e,RuntimeError):r['error']=str(e)
            r['elapsed_ms']=(time.perf_counter()-t)*1000;records.append(r)
            write(out/(backend+'.json'),{'backend':backend,'metadata':metadata,'summary':summarize(records),'records':records})
            if 'error' in r:
                print(backend, 'case',i, r['error'],flush=True)
                if backend=='jev':break # No repeated charges/retries on uncertain failure.
        if client:client.close()
        print(backend,json.dumps(summarize(records)),flush=True)
        if backend!='jev':
            engine_mlx._model=engine_mlx._tokenizer=None
    summary={}
    for name in ['upstream','ours_base','ours_adapter','jev']:
        p=out/(name+'.json')
        if p.exists():
            data=json.loads(p.read_text())
            if data['metadata']['dataset_sha256']!=digest:raise ValueError('Dataset mismatch; choose a new output folder')
            summary[name]=data['summary']
    write(out/'summary.json',summary)

if __name__=='__main__':main()
