"""Train the balanced, order-robust banking v4 candidate-logit adapter."""
from __future__ import annotations
import argparse, json, math, random, time
from collections import defaultdict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from banking_v4_data import save as save_data
from nimble_1_5b_mlx import ROOT, LORA_KEYS, candidate_logits, dump, encode, evaluate, schedule, sha

def pair_order(rows,seed):
    groups=defaultdict(list)
    for row in rows:groups[row['pair']].append(row)
    if any(len(v)!=2 for v in groups.values()):raise ValueError('Every family must be a pair')
    pairs=list(groups.values());random.Random(seed).shuffle(pairs);return [row for pair in pairs for row in pair]

def position_share(summary):
    c=summary['selected_position_counts'];return max(c.values())/sum(c.values())

def fit_temperature(records):
    best=(float('inf'),1.)
    for i in range(20,501):
        t=i/100;nll=0.
        for row in records:
            weights=[max(p,1e-15)**(1/t) for p in row['probabilities']];z=sum(weights)
            nll-=math.log(max(weights[row['label']]/z,1e-15))
        score=nll/len(records)
        if score<best[0]:best=(score,t)
    return best[1],best[0]

def calibrated(records,t):
    out=[]
    for row in records:
        item=dict(row);w=[max(p,1e-15)**(1/t) for p in row['probabilities']];z=sum(w);p=[x/z for x in w];pred=max(range(len(p)),key=p.__getitem__);gold=row['label']
        item.update(probabilities=p,prediction=pred,correct=pred==gold,confidence=p[pred],reference_probability=p[gold],nll=-math.log(max(p[gold],1e-15)),brier=sum((x-int(i==gold))**2 for i,x in enumerate(p)))
        out.append(item)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/qwen-1.5b-4bit');ap.add_argument('--out',default='runs/banking-v4')
    ap.add_argument('--rank',type=int,default=8);ap.add_argument('--layers',type=int,default=8);ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--accum',type=int,default=8);ap.add_argument('--seed',type=int,default=29);ap.add_argument('--max-tokens',type=int,default=1536)
    ap.add_argument('--steps',type=int,default=96);args=ap.parse_args();out=ROOT/args.out
    if (out/'report.json').exists():ap.error('completed output exists; choose a new --out')
    raw,manifest=save_data(out/'data');mx.random.seed(args.seed);mx.set_cache_limit(1024*1024**2);model,tok=load(str(ROOT/args.model));model.eval()
    train=encode(raw['train'],tok,args.max_tokens,True,args.seed)
    val=[encode(raw['validation'],tok,args.max_tokens,False,s) for s in (None,args.seed,args.seed+1)]
    test=[encode(raw['test'],tok,args.max_tokens,False,s) for s in (None,args.seed)]
    total=args.steps;warmup=max(1,round(total*.1));out.mkdir(parents=True,exist_ok=True)
    contract={'task':'banking_balanced_candidate_classification_v4','model':'mlx-community/Qwen2.5-1.5B-Instruct-4bit','rank':args.rank,'layers':args.layers,
      'targets':LORA_KEYS,'seed':args.seed,'learning_rate':args.lr,'effective_batch':args.accum,'steps':total,'warmup_steps':warmup,'max_tokens':args.max_tokens,
      'pair_aware_batches':True,'candidate_order_training':f'seeded per record {args.seed}','validation_orders':['original',args.seed,args.seed+1],
      'prompt_code_sha256':sha(ROOT/'reference-nimble/nimble/scoring/parallel_schema.py'),'dataset':manifest,
      'selection':'minimum accuracy and pair accuracy over three validation orders, with ECE penalty; JevBench excluded'};dump(out/'schema_config.json',contract)
    base=[]
    for rows in val:
        _,summary,_=evaluate(model,rows);base.append(summary)
    dump(out/'base_validation.json',base)
    model.freeze();lora={'rank':args.rank,'scale':2*args.rank,'dropout':.05,'keys':LORA_KEYS};linear_to_lora_layers(model,args.layers,lora)
    trainable=dict(tree_flatten(model.trainable_parameters()));config={'fine_tune_type':'lora','num_layers':args.layers,'lora_parameters':lora};dump(out/'adapter_config.json',config)
    best=out/'best';best.mkdir(exist_ok=True);dump(best/'adapter_config.json',config);mx.save_safetensors(str(best/'adapters.safetensors'),trainable)
    opt=optim.AdamW(learning_rate=args.lr,weight_decay=0.01)
    def loss_fn(m,row):
        z=candidate_logits(m,row);smooth=.02;n=len(row['candidate_ids']);target=(1-smooth)*(-z[row['label']]+mx.logsumexp(z));uniform=smooth*(-mx.mean(z)+mx.logsumexp(z));return target+uniform
    vg=nn.value_and_grad(model,loss_fn);ordered=pair_order(train,args.seed);pos=0;history=[];best_score=-1e9;best_step=0;compute=0.;started=time.perf_counter()
    for step in range(1,total+1):
        model.train();ga=None;ls=0.;tick=time.perf_counter()
        for _ in range(args.accum):
            if pos>=len(ordered):ordered=pair_order(train,args.seed+step);pos=0
            loss,grad=vg(model,ordered[pos]);pos+=1;mx.eval(loss,grad);ga=grad if ga is None else tree_map(lambda a,b:a+b,ga,grad);mx.eval(ga);ls+=float(loss)
        grads=tree_map(lambda g:g/args.accum,ga);grads,norm=optim.clip_grad_norm(grads,1.);lr=schedule(step,total,warmup,args.lr);opt.learning_rate=lr;opt.update(model,grads);mx.eval(model.parameters(),opt.state);compute+=time.perf_counter()-tick
        entry={'step':step,'loss':ls/args.accum,'gradient_norm':float(norm),'learning_rate':lr}
        if step%24==0 or step==total:
            model.eval();summaries=[]
            for rows in val:
                _,summary,_=evaluate(model,rows);summaries.append(summary)
            min_acc=min(s['by_kind']['all']['accuracy'] for s in summaries);min_pair=min(s['contrastive_pair_accuracy'] for s in summaries);max_ece=max(s['ece_10_bins'] for s in summaries);max_pos=max(position_share(s) for s in summaries)
            accepted=min_acc>max(s['by_kind']['all']['accuracy'] for s in base) and max_pos<.75
            score=min_acc+min_pair-.25*max_ece;entry.update(validation=summaries,accepted=accepted,selection_score=score)
            ck=out/f'checkpoint-{step}';ck.mkdir(exist_ok=True);dump(ck/'adapter_config.json',config);weights=dict(tree_flatten(model.trainable_parameters()));mx.save_safetensors(str(ck/'adapters.safetensors'),weights)
            if accepted and score>best_score:best_score=score;best_step=step;mx.save_safetensors(str(best/'adapters.safetensors'),weights)
            print(json.dumps({'step':step,'loss':entry['loss'],'accepted':accepted,'min_accuracy':min_acc,'min_pair':min_pair,'max_ece':max_ece,'max_position_share':max_pos}),flush=True)
        history.append(entry);dump(out/'history.json',history)
    if not best_step:raise RuntimeError('No checkpoint passed validation gates')
    model.eval();model.load_weights(str(best/'adapters.safetensors'),strict=False);val_records=[];val_summaries=[]
    for rows in val:
        records,summary,_=evaluate(model,rows);val_records.append(records);val_summaries.append(summary)
    temperature,nll=fit_temperature(val_records[0]);dump(best/'calibration.json',{'temperature':temperature,'fit_split':'validation original order','objective':'categorical NLL','nll':nll})
    test_out=[]
    for rows in test:
        records,summary,seconds=evaluate(model,rows);cal=calibrated(records,temperature);test_out.append({'summary':summary,'calibrated_summary':__import__('nimble_1_5b_mlx').summarize(cal),'seconds':seconds,'rows':records})
    dump(out/'test.json',test_out)
    report={'contract':contract,'selected_step':best_step,'validation':val_summaries,'temperature':temperature,'test':[{'raw':x['summary'],'calibrated':x['calibrated_summary']} for x in test_out],
      'training_compute_seconds':compute,'wall_seconds':time.perf_counter()-started,'limitations':['Synthetic deterministic banking policies','Single seed','No human-reviewed test','Calibration fitted only on synthetic validation','JevBench excluded from training and selection']}
    dump(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
