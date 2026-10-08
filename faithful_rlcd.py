"""Parity-first wrapper/trainer for the unchanged community Qwen-RLCD MLX engine."""
import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from core import engine_mlx
from core.schema import StructuredSchema
from mlx_jev_repro import SCHEMA, LORA, DecisionEngine
from banking_data import save as save_data


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')


def schema_for():
    return StructuredSchema({s['name']:{'type':'boolean' if s['type']=='noul' else 'enum',
        'description':s['question'],'choices':None if s['type']=='noul' else [str(c) for c in s['choices']]} for s in SCHEMA})


def canonical(raw):
    result=[]
    for s in SCHEMA:
        values={v['choice']:v['probability'] for v in raw['field_telemetry'][s['name']]['top_choices']}
        p=np.array([values[str(c).lower() if isinstance(c,bool) else str(c)] for c in s['choices']],dtype=float)
        assert np.isfinite(p).all() and np.min(p)>=0 and abs(p.sum()-1)<.002
        result.append(p/p.sum())
    return result


class Faithful:
    def __init__(self,model,tokenizer):
        self.model,self.tokenizer=model,tokenizer
        self.schema=schema_for()
        self.meta=self.schema.compile_parallel_metadata(tokenizer)
        if any(self.meta['has_collisions']):
            raise ValueError('Training supports only distinct upstream candidate tokens; collision fallback is heuristic')
        self.ids=[]
        for s,c in zip(SCHEMA,self.meta['cands_per_field']):
            self.ids.append(mx.array(c[::-1] if s['type']=='noul' else c))

    def prefix(self,state):
        # Exact upstream template and tokenizer call, including final newline.
        text=(f'<|im_start|>system\nClassify JSON attributes:\n{self.schema.to_parallel_schema_str()}<|im_end|>\n'
              f'<|im_start|>user\n{state}<|im_end|>\n<|im_start|>assistant\n{{\n')
        ids=self.tokenizer.encode(text)
        if len(ids)+max(self.meta['suffix_lengths'])>384:raise ValueError('Prompt exceeds 384-token cap')
        return mx.array([ids])

    def raw(self,state,temperature=1.):
        # Public inference uses the original implementation, including its native return values.
        engine_mlx._model,engine_mlx._tokenizer=self.model,self.tokenizer
        return engine_mlx.run_parallel_generation(state,self.schema,temperature=temperature)

    def differentiable(self,prefix):
        # Differentiable twin of upstream's collision-free path. No detached prefix cache.
        cache=make_prompt_cache(self.model)
        self.model(prefix,cache=cache)
        branches=[]
        for c in cache:
            nc=copy.copy(c)
            nc.keys=mx.repeat(c.keys,len(SCHEMA),axis=0)
            nc.values=mx.repeat(c.values,len(SCHEMA),axis=0)
            branches.append(nc)
        logits=self.model(self.meta['suffixes_batch'],cache=branches)
        return [logits[i,self.meta['suffix_lengths'][i]-1,ids].astype(mx.float32) for i,ids in enumerate(self.ids)]

    def probabilities(self,state):
        return canonical(self.raw(state))


def evaluate(engine,rows):
    records=[]
    for row in rows:
        t=time.perf_counter();ps=engine.probabilities(row['state'])
        records.append({'id':row['id'],'group':row['group'],'labels':row['labels'],
                        'probabilities':[np.asarray(p).tolist() for p in ps],
                        'elapsed_ms':(time.perf_counter()-t)*1000})
    return records


def transform(records,t):
    out=copy.deepcopy(records)
    for row in out:
        for i,p in enumerate(row['probabilities']):
            z=np.log(np.maximum(p,1e-12))/t;v=np.exp(z-z.max());row['probabilities'][i]=(v/v.sum()).tolist()
    return out


def metrics(records):
    acc=[];nll=[];brier=[];conf=[];fields=[[],[],[]];groups={}
    for row in records:
        rc=[]
        for i,(p,y) in enumerate(zip(row['probabilities'],row['labels'])):
            p=np.array(p);c=int(p.argmax())==y
            acc.append(c);rc.append(c);conf.append(p.max());fields[i].append(c)
            nll.append(-math.log(max(float(p[y]),1e-12)))
            brier.append(float(np.sum((p-np.eye(len(p))[y])**2)))
        groups.setdefault(row['group'],[]).extend(rc)
    acc=np.array(acc);conf=np.array(conf);ece=0.
    for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        m=(conf>low)&(conf<=high)
        if m.any():ece+=float(m.mean()*abs(acc[m].mean()-conf[m].mean()))
    return {'cases':len(records),'groups':len(groups),'decisions':len(acc),'correct':int(acc.sum()),
        'accuracy':float(acc.mean()),'nll':float(np.mean(nll)),'brier':float(np.mean(brier)),
        'ece_5_bins':ece,'field_accuracy':{s['name']:float(np.mean(a)) for s,a in zip(SCHEMA,fields)},
        'median_ms':float(np.median([r['elapsed_ms'] for r in records]))}


def parity(engine,rows):
    # Observe actual upstream model calls, rather than comparing two handmade prompt strings.
    class Spy:
        def __init__(self,m):self.m=m;self.inputs=[]
        def __getattr__(self,k):return getattr(self.m,k)
        def __call__(self,x,*a,**kw):self.inputs.append(x.tolist());return self.m(x,*a,**kw)
    errors=[]
    for row in rows:
        spy=Spy(engine.model)
        engine_mlx._model,engine_mlx._tokenizer=spy,engine.tokenizer
        raw=engine_mlx.run_parallel_generation(row['state'],engine.schema)
        assert len(spy.inputs)==2, 'Unexpected candidate continuation'
        assert spy.inputs[0]==engine.prefix(row['state']).tolist()
        assert spy.inputs[1]==engine.meta['suffixes_batch'].tolist()
        direct=canonical(raw)
        twin=[np.array(mx.softmax(z).tolist()) for z in engine.differentiable(engine.prefix(row['state']))]
        error=max(float(np.max(np.abs(a-b))) for a,b in zip(direct,twin));errors.append(error)
        assert error<.0002, ('Training/inference probability mismatch',error)
        wrapped=engine.raw(row['state'])
        assert wrapped['parsed_json']==raw['parsed_json']
        assert wrapped['field_telemetry']==raw['field_telemetry']
    return {'cases':len(rows),'prefix_and_suffix_tokens_exact':True,'wrapper_answers_and_probabilities_exact':True,
            'max_differentiable_probability_error':max(errors),'tolerance':.0002}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='models/qwen-1.5b-4bit')
    ap.add_argument('--out',default='runs/faithful-v1')
    ap.add_argument('--adapter')
    ap.add_argument('--state')
    ap.add_argument('--steps',type=int,default=80)
    ap.add_argument('--accum',type=int,default=4)
    ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--seed',type=int,default=17)
    args=ap.parse_args()
    if args.steps<1 or args.accum<1 or args.lr<=0:ap.error('steps, accum and lr must be positive')
    out=ROOT/args.out
    if not args.state and (out/'report.json').exists():ap.error('Experiment already completed; choose a new --out')
    mx.random.seed(args.seed);mx.set_cache_limit(512*1024**2)
    model,tok=load(str(ROOT/args.model),adapter_path=str(ROOT/args.adapter) if args.adapter else None)
    model.eval();engine=Faithful(model,tok)
    if args.state:
        temperature=1.
        if args.adapter and (ROOT/args.adapter/'temperature.json').exists():
            temperature=json.loads((ROOT/args.adapter/'temperature.json').read_text())['temperature']
        print(json.dumps(engine.raw(args.state,temperature),indent=2));return
    if args.adapter:ap.error('Experiments start from base weights; --adapter is for inference')
    data,manifest=save_data(out/'data')
    report={'config':vars(args),'schema':SCHEMA,'datasets':manifest}
    start=time.perf_counter()
    # Data frozen before the first model evaluation. No test-set selection.
    print('Checking exact upstream parity...',flush=True)
    report['parity_before']=parity(engine,data['validation'][::3])
    print(json.dumps(report['parity_before']),flush=True)
    print('Evaluating fixed base/validation...',flush=True)
    base_val=evaluate(engine,data['validation']);base_test=evaluate(engine,data['test'])
    dump(out/'upstream_base_test.json',base_test)
    dump(out/'upstream_base_validation.json',base_val)
    report['upstream_base']=metrics(base_test)
    print('Evaluating previous letter scorer for reference...',flush=True)
    old=DecisionEngine(model,tok,384)
    class Old:
        def probabilities(self,state):return [np.array(mx.softmax(z).tolist()) for z in old.parallel_logits(state)]
    old_records=evaluate(Old(),data['test']);dump(out/'old_letter_base_test.json',old_records)
    report['old_letter_base']=metrics(old_records)
    model.freeze();linear_to_lora_layers(model,4,LORA)
    trainable=dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all('lora_' in k for k in trainable)
    config={'fine_tune_type':'lora','num_layers':4,'lora_parameters':LORA}
    best=out/'best';best.mkdir(exist_ok=True)
    dump(best/'adapter_config.json',config)
    mx.save_safetensors(str(best/'adapters.safetensors'),trainable)
    baseline_adapter=evaluate(engine,data['validation'])
    best_nll=metrics(baseline_adapter)['nll'];best_step=0
    # Zero LoRA is a candidate, so worsening training can select the unchanged model.
    best_val=baseline_adapter
    history=[{'step':0,'validation':metrics(baseline_adapter)}]
    prefixes={r['id']:engine.prefix(r['state']) for r in data['train']}
    optimizer=optim.Adam(learning_rate=args.lr)
    def loss_fn(m,x,ys):
        zs=engine.differentiable(x);loss=mx.array(0.,dtype=mx.float32)
        for z,y in zip(zs,ys):
            logp=z-mx.logsumexp(z);p=mx.exp(logp)
            target=(mx.arange(len(z))==y).astype(mx.float32)
            loss+=-logp[y]+.5*mx.sum((p-target)**2)
        return loss/len(zs)
    vg=nn.value_and_grad(model,loss_fn)
    rng=random.Random(args.seed);order=list(data['train']);rng.shuffle(order);position=0
    train_start=time.perf_counter();training_compute=0.
    print('Training: exact candidate tokens, CE + half Brier, no RL...',flush=True)
    for step in range(1,args.steps+1):
        model.train();grad_acc=None;loss_sum=0.;t0=time.perf_counter()
        for _ in range(args.accum):
            if position==len(order):rng.shuffle(order);position=0
            r=order[position];position+=1
            loss,grad=vg(model,prefixes[r['id']],r['labels'])
            mx.eval(loss,grad)
            if not math.isfinite(float(loss)):raise RuntimeError('Nonfinite training loss')
            grad_acc=grad if grad_acc is None else tree_map(lambda a,b:a+b,grad_acc,grad)
            mx.eval(grad_acc);loss_sum+=float(loss)
        grads=tree_map(lambda g:g/args.accum,grad_acc)
        grads,norm=optim.clip_grad_norm(grads,1.)
        assert math.isfinite(float(norm))
        optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state)
        training_compute+=time.perf_counter()-t0
        entry={'step':step,'loss':loss_sum/args.accum,'gradient_norm':float(norm)}
        if step%20==0 or step==args.steps:
            model.eval();val=evaluate(engine,data['validation']);entry['validation']=metrics(val)
            checkpoint=out/f'checkpoint-{step}';checkpoint.mkdir(exist_ok=True)
            dump(checkpoint/'adapter_config.json',config)
            weights=dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(checkpoint/'adapters.safetensors'),weights)
            if entry['validation']['nll']<best_nll:
                best_nll=entry['validation']['nll'];best_step=step;best_val=val
                mx.save_safetensors(str(best/'adapters.safetensors'),weights)
            print(json.dumps(entry),flush=True)
        elif step%5==0:print(json.dumps(entry),flush=True)
        history.append(entry);dump(out/'history.json',history)
    model.eval();model.load_weights(str(best/'adapters.safetensors'),strict=False)
    print('Evaluating validation-selected checkpoint on frozen test...',flush=True)
    report['parity_after']=parity(engine,data['validation'][::3])
    trained=evaluate(engine,data['test']);dump(out/'trained_test.json',trained)
    dump(out/'selected_validation.json',best_val)
    report['trained']=metrics(trained)
    # Fit on validation, deploy through upstream's temperature argument.
    # Grid evaluation uses saved rounded upstream probabilities; final deployment is re-evaluated.
    grid=np.geomspace(.5,4.,61)
    temperature=float(min(grid,key=lambda t:metrics(transform(best_val,t))['nll']))
    dump(best/'temperature.json',{'temperature':temperature,'fit_split':'validation','criterion':'NLL','grid':[.5,4.,61]})
    class Calibrated:
        def probabilities(self,state):return canonical(engine.raw(state,temperature))
    calibrated=evaluate(Calibrated(),data['test']);dump(out/'trained_temperature_test.json',calibrated)
    report['trained_temperature']=metrics(calibrated)
    report['selection']={'best_step':best_step,'validation_nll':best_nll,'temperature':temperature,'test_used_for_selection':False}
    report['training']={'updates':args.steps,'accumulation':args.accum,'trainable_parameters':sum(v.size for v in trainable.values()),'compute_seconds':training_compute,'including_validation_seconds':time.perf_counter()-train_start}
    report['wall_seconds']=time.perf_counter()-start
    report['peak_mlx_gb']=mx.get_peak_memory()/1e9
    report['upstream_commit']=subprocess.check_output(['git','-C',str(ROOT/'upstream'),'rev-parse','HEAD'],text=True).strip()
    report['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['faithful_rlcd.py','banking_data.py','upstream/core/engine_mlx.py','upstream/core/schema.py']}
    report['caveat']='One seed, hand-authored synthetic rubric, correlated wording variants; no production or general calibration claim.'
    dump(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
