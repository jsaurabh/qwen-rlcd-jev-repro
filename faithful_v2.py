"""Compositional banking adapter on the unchanged public Qwen-RLCD MLX path."""
import argparse, copy, json, math, random, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'upstream'))
from core import engine_mlx
from core.schema import StructuredSchema
from mlx_jev_repro import LORA
from banking_v2_data import FIELDS, save as save_data

WEIGHTS={"department":1.,"suspected_fraud":1.5,"disputed_authorization":1.25,"conflicting_information":1.,"policy_exception":1.,"ongoing_loss":2.,"customer_blocked":1.25,"deadline_pressure":1.}
CRITICAL=("suspected_fraud","ongoing_loss","customer_blocked")
POSITIVE_EXAMPLE_WEIGHT={"suspected_fraud":4.,"disputed_authorization":2.,"ongoing_loss":2.}

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')

def schema_for():
    return StructuredSchema({s['name']:{'type':'boolean' if s['type']=='noul' else 'enum','description':s['question'],'choices':None if s['type']=='noul' else s['choices']} for s in FIELDS})

def canonical(raw):
    out=[]
    for s in FIELDS:
        values={v['choice']:v['probability'] for v in raw['field_telemetry'][s['name']]['top_choices']};choices=s.get('choices',[False,True])
        p=np.array([values[str(c).lower() if isinstance(c,bool) else str(c)] for c in choices],dtype=float)
        assert np.isfinite(p).all() and np.min(p)>=0 and abs(p.sum()-1)<.002;out.append(p/p.sum())
    return out

class Engine:
    def __init__(self,model,tokenizer,max_tokens=512):
        self.model,self.tokenizer,self.max_tokens=model,tokenizer,max_tokens;self.schema=schema_for();self.meta=self.schema.compile_parallel_metadata(tokenizer)
        if any(self.meta['has_collisions']):raise ValueError('V2 requires distinct first-token candidates')
        self.ids=[mx.array(c[::-1] if s['type']=='noul' else c) for s,c in zip(FIELDS,self.meta['cands_per_field'])]
    def prefix(self,state):
        text=f'<|im_start|>system\nClassify JSON attributes:\n{self.schema.to_parallel_schema_str()}<|im_end|>\n<|im_start|>user\n{state}<|im_end|>\n<|im_start|>assistant\n{{\n'
        ids=self.tokenizer.encode(text)
        if len(ids)+max(self.meta['suffix_lengths'])>self.max_tokens:raise ValueError(f'Prompt exceeds {self.max_tokens}-token cap')
        return mx.array([ids])
    def raw(self,state,temperature=1.):
        engine_mlx._model,engine_mlx._tokenizer=self.model,self.tokenizer;return engine_mlx.run_parallel_generation(state,self.schema,temperature=temperature)
    def differentiable(self,prefix):
        cache=make_prompt_cache(self.model);self.model(prefix,cache=cache);branches=[]
        for c in cache:
            nc=copy.copy(c);nc.keys=mx.repeat(c.keys,len(FIELDS),axis=0);nc.values=mx.repeat(c.values,len(FIELDS),axis=0);branches.append(nc)
        logits=self.model(self.meta['suffixes_batch'],cache=branches)
        return [logits[i,self.meta['suffix_lengths'][i]-1,ids].astype(mx.float32) for i,ids in enumerate(self.ids)]
    def probabilities(self,state):return canonical(self.raw(state))

def derived(probabilities):
    p={s['name']:np.asarray(v,dtype=float) for s,v in zip(FIELDS,probabilities)}
    review=1-float(np.prod([1-p[n][1] for n in ('suspected_fraud','disputed_authorization','conflicting_information','policy_exception')]))
    loss=float(p['ongoing_loss'][1]);blocked=float(p['customer_blocked'][1]);deadline=p['deadline_pressure']
    immediate=1-(1-loss)*(1-float(deadline[2]));routine=(1-loss)*(1-blocked)*float(deadline[0]);urgency=np.array([routine,max(0.,1-routine-immediate),immediate]);urgency/=urgency.sum()
    return {'needs_review':[1-review,review],'urgency':urgency.tolist()}

def evaluate(engine,rows):
    out=[]
    for row in rows:
        start=time.perf_counter();ps=engine.probabilities(row['state']);out.append({'id':row['id'],'group':row['group'],'labels':row['labels'],'targets':row['targets'],'probabilities':[v.tolist() for v in ps],'derived':derived(ps),'elapsed_ms':(time.perf_counter()-start)*1000})
    return out

def hard_derived(labels):return int(any(labels[i] for i in (1,2,3,4))),max(2 if labels[5] else 0,1 if labels[6] else 0,labels[7])

def metrics(records):
    per={s['name']:{'correct':0,'n':0,'nll':[],'brier':[],'tp':0,'fn':0} for s in FIELDS};confidences=[];corrects=[];cost=0.;dc={'needs_review':0,'urgency':0}
    for row in records:
        for s,p,y,target in zip(FIELDS,row['probabilities'],row['labels'],row['targets']):
            p=np.asarray(p);pred=int(p.argmax());m=per[s['name']];m['correct']+=pred==y;m['n']+=1;m['nll'].append(-float(np.sum(np.asarray(target)*np.log(np.maximum(p,1e-12)))));m['brier'].append(float(np.sum((p-np.asarray(target))**2)));confidences.append(float(p.max()));corrects.append(pred==y)
            if len(p)==2 and y==1:m['tp']+=pred==1;m['fn']+=pred==0
        gr,gu=hard_derived(row['labels']);pr=int(np.argmax(row['derived']['needs_review']));pu=int(np.argmax(row['derived']['urgency']));dc['needs_review']+=pr==gr;dc['urgency']+=pu==gu
        cost+=10*(row['labels'][5]==1 and np.argmax(row['probabilities'][5])==0)+5*(gr==1 and pr==0)+1*(gr==0 and pr==1)+2*(np.argmax(row['probabilities'][0])!=row['labels'][0])+4*max(0,gu-pu)
    acc=np.asarray(corrects);conf=np.asarray(confidences);ece=0.
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(conf>lo)&(conf<=hi)
        if mask.any():ece+=float(mask.mean()*abs(acc[mask].mean()-conf[mask].mean()))
    field={};weighted=den=0.
    for name,m in per.items():
        field[name]={'accuracy':m['correct']/m['n'],'nll':float(np.mean(m['nll'])),'brier':float(np.mean(m['brier']))}
        if m['tp']+m['fn']:field[name]['positive_recall']=m['tp']/(m['tp']+m['fn'])
        weighted+=WEIGHTS[name]*field[name]['nll'];den+=WEIGHTS[name]
    return {'cases':len(records),'decisions':len(corrects),'accuracy':float(acc.mean()),'weighted_nll':weighted/den,'ece_5_bins':ece,'field':field,'derived_accuracy':{k:v/len(records) for k,v in dc.items()},'operational_cost_per_case':cost/len(records),'median_ms':float(np.median([r['elapsed_ms'] for r in records]))}

def acceptable(candidate,baseline):
    reasons=[]
    for name in CRITICAL:
        ca=candidate['field'][name]['accuracy'];ba=baseline['field'][name]['accuracy'];cr=candidate['field'][name].get('positive_recall',1.);br=baseline['field'][name].get('positive_recall',1.)
        if ca+0.05<ba:reasons.append(f'{name} accuracy {ca:.3f} below floor {ba-.05:.3f}')
        if cr+0.02<br:reasons.append(f'{name} recall {cr:.3f} below floor {br-.02:.3f}')
    return not reasons,reasons

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/qwen-1.5b-4bit');ap.add_argument('--out',default='runs/compositional-v2');ap.add_argument('--adapter');ap.add_argument('--state');ap.add_argument('--steps',type=int,default=100);ap.add_argument('--accum',type=int,default=4);ap.add_argument('--lr',type=float,default=4e-5);ap.add_argument('--seed',type=int,default=29);args=ap.parse_args()
    if args.steps<1 or args.accum<1 or args.lr<=0:ap.error('steps, accum and lr must be positive')
    out=ROOT/args.out
    if not args.state and (out/'report.json').exists():ap.error('Experiment already completed; choose a new --out')
    mx.random.seed(args.seed);mx.set_cache_limit(512*1024**2);model,tok=load(str(ROOT/args.model),adapter_path=str(ROOT/args.adapter) if args.adapter else None);model.eval();engine=Engine(model,tok)
    if args.state:
        raw=engine.raw(args.state);print(json.dumps({'raw':raw,'derived':derived(canonical(raw))},indent=2));return
    if args.adapter:ap.error('--adapter is for inference only')
    data,manifest=save_data(out/'data');report={'config':vars(args),'schema':FIELDS,'loss_weights':WEIGHTS,'positive_example_weights':POSITIVE_EXAMPLE_WEIGHT,'datasets':manifest}
    print('Evaluating frozen base validation and test sets...',flush=True);base_val=evaluate(engine,data['validation']);base_test=evaluate(engine,data['test']);dump(out/'base_validation.json',base_val);dump(out/'base_test.json',base_test);base_metrics=metrics(base_val);report['base_validation']=base_metrics;report['base_test']=metrics(base_test)
    model.freeze();linear_to_lora_layers(model,4,LORA);trainable=dict(tree_flatten(model.trainable_parameters()));assert trainable and all('lora_' in k for k in trainable)
    config={'fine_tune_type':'lora','num_layers':4,'lora_parameters':LORA,'schema_version':'banking-compositional-v2'};best=out/'best';best.mkdir(parents=True,exist_ok=True);dump(best/'adapter_config.json',config);mx.save_safetensors(str(best/'adapters.safetensors'),trainable)
    best_score=base_metrics['weighted_nll'];best_step=0;best_val=base_val;history=[{'step':0,'accepted':True,'validation':base_metrics}];prefixes={r['id']:engine.prefix(r['state']) for r in data['train']};optimizer=optim.Adam(learning_rate=args.lr)
    def loss_fn(m,x,targets):
        total=mx.array(0.,dtype=mx.float32);den=0.
        for s,z,target in zip(FIELDS,engine.differentiable(x),targets):
            logp=z-mx.logsumexp(z);p=mx.exp(logp);t=mx.array(target,dtype=mx.float32);w=WEIGHTS[s['name']]
            # False negatives carry more operational cost. Weight the whole example so
            # the target distribution remains a calibrated soft label.
            positive_weight=POSITIVE_EXAMPLE_WEIGHT.get(s['name'],1.)
            example_weight=1+(positive_weight-1)*t[1] if len(target)==2 else 1.
            total+=w*example_weight*(-mx.sum(t*logp)+.5*mx.sum((p-t)**2));den+=w*example_weight
        return total/den
    vg=nn.value_and_grad(model,loss_fn);rng=random.Random(args.seed);order=list(data['train']);rng.shuffle(order);pos=0;compute=0.;wall=time.perf_counter();print('Training calibrated atomic judgments with weighted CE + Brier...',flush=True)
    for step in range(1,args.steps+1):
        model.train();grad_acc=None;loss_sum=0.;tick=time.perf_counter()
        for _ in range(args.accum):
            if pos==len(order):rng.shuffle(order);pos=0
            row=order[pos];pos+=1;loss,grad=vg(model,prefixes[row['id']],row['targets']);mx.eval(loss,grad)
            if not math.isfinite(float(loss)):raise RuntimeError('Nonfinite loss')
            grad_acc=grad if grad_acc is None else tree_map(lambda a,b:a+b,grad_acc,grad);mx.eval(grad_acc);loss_sum+=float(loss)
        grads=tree_map(lambda g:g/args.accum,grad_acc);grads,norm=optim.clip_grad_norm(grads,1.);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state);compute+=time.perf_counter()-tick;entry={'step':step,'loss':loss_sum/args.accum,'gradient_norm':float(norm)}
        if step%20==0 or step==args.steps:
            model.eval();val=evaluate(engine,data['validation']);vm=metrics(val);ok,reasons=acceptable(vm,base_metrics);entry.update({'validation':vm,'accepted':ok,'rejection_reasons':reasons});checkpoint=out/f'checkpoint-{step}';checkpoint.mkdir(exist_ok=True);dump(checkpoint/'adapter_config.json',config);weights=dict(tree_flatten(model.trainable_parameters()));mx.save_safetensors(str(checkpoint/'adapters.safetensors'),weights)
            if ok and vm['weighted_nll']<best_score:best_score=vm['weighted_nll'];best_step=step;best_val=val;mx.save_safetensors(str(best/'adapters.safetensors'),weights)
            print(json.dumps({'step':step,'loss':entry['loss'],'accepted':ok,'weighted_nll':vm['weighted_nll'],'cost':vm['operational_cost_per_case'],'reasons':reasons}),flush=True)
        elif step%5==0:print(json.dumps(entry),flush=True)
        history.append(entry);dump(out/'history.json',history)
    model.eval();model.load_weights(str(best/'adapters.safetensors'),strict=False);trained=evaluate(engine,data['test']);dump(out/'trained_test.json',trained);dump(out/'selected_validation.json',best_val);report['selected_step']=best_step;report['trained_test']=metrics(trained);report['training_compute_seconds']=compute;report['wall_seconds']=time.perf_counter()-wall;report['selection_rule']={'objective':'weighted validation NLL','critical_fields':list(CRITICAL),'accuracy_floor':'base - 0.05','positive_recall_floor':'base - 0.02','test_used_for_selection':False};dump(out/'report.json',report);print(json.dumps({'selected_step':best_step,'base_test':report['base_test'],'trained_test':report['trained_test'],'report':str(out/'report.json')},indent=2),flush=True)

if __name__=='__main__':main()
