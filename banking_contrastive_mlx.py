"""Pair-aware banking contrastive training with validation and position-bias gates."""
import argparse, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from banking_contrastive_data import save as save_data
from nimble_1_5b_mlx import ROOT, LORA_KEYS, candidate_logits, dump, encode, evaluate, schedule, sha

def pair_order(rows,seed):
    groups=defaultdict(list)
    for r in rows:groups[r['pair']].append(r)
    if any(len(v)!=2 for v in groups.values()):raise ValueError('Every training family must be one complete pair')
    pairs=list(groups.values());random.Random(seed).shuffle(pairs);return [r for pair in pairs for r in pair]

def position_share(summary):
    counts=summary['selected_position_counts'];return max(counts.values())/sum(counts.values())

def accepted(original,permuted,base_original,base_permuted):
    accuracy=min(original['by_kind']['all']['accuracy'],permuted['by_kind']['all']['accuracy'])
    base_accuracy=min(base_original['by_kind']['all']['accuracy'],base_permuted['by_kind']['all']['accuracy'])
    family=min(original['contrastive_family_accuracy'],permuted['contrastive_family_accuracy'])
    base_family=min(base_original['contrastive_family_accuracy'],base_permuted['contrastive_family_accuracy'])
    reasons=[]
    if accuracy<=base_accuracy:reasons.append(f'worst-order accuracy {accuracy:.3f} <= base {base_accuracy:.3f}')
    if family<=base_family:reasons.append(f'worst-order pair accuracy {family:.3f} <= base {base_family:.3f}')
    if max(position_share(original),position_share(permuted))>.80:reasons.append('candidate-position concentration exceeds 80%')
    return not reasons,reasons,accuracy+family

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/qwen-1.5b-4bit');ap.add_argument('--out',default='runs/banking-contrastive-v3')
    ap.add_argument('--rank',type=int,default=8);ap.add_argument('--layers',type=int,default=8);ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--accum',type=int,default=8);ap.add_argument('--seed',type=int,default=17);ap.add_argument('--max-tokens',type=int,default=768)
    ap.add_argument('--steps',type=int);args=ap.parse_args();out=ROOT/args.out
    if (out/'report.json').exists():ap.error('completed output exists; choose a new --out')
    raw,manifest=save_data(out/'data');mx.random.seed(args.seed);mx.set_cache_limit(1024*1024**2)
    model,tok=load(str(ROOT/args.model));model.eval();train=encode(raw['train'],tok,args.max_tokens,True,args.seed)
    validation=encode(raw['validation'],tok,args.max_tokens,False);validation_shuffled=encode(raw['validation'],tok,args.max_tokens,False,args.seed)
    test=encode(raw['test'],tok,args.max_tokens,False);test_shuffled=encode(raw['test'],tok,args.max_tokens,False,args.seed)
    total=args.steps or math.ceil(len(train)/args.accum);warmup=max(1,round(total*.1));out.mkdir(parents=True,exist_ok=True)
    contract={'task':'banking_contrastive_candidate_classification_v1','model':'mlx-community/Qwen2.5-1.5B-Instruct-4bit','rank':args.rank,'layers':args.layers,
              'targets':LORA_KEYS,'seed':args.seed,'learning_rate':args.lr,'effective_batch':args.accum,'steps':total,'warmup_steps':warmup,
              'max_tokens':args.max_tokens,'pair_aware_batches':True,'prompt_code_sha256':sha(ROOT/'reference-nimble/nimble/scoring/parallel_schema.py'),
              'dataset':manifest,'selection':'validation original+seeded candidate permutation; test untouched'};dump(out/'schema_config.json',contract)
    print('Evaluating base validation in two candidate orders...',flush=True)
    base_val,bvo,_=evaluate(model,validation);base_perm,bvp,_=evaluate(model,validation_shuffled);dump(out/'base_validation.json',{'original':base_val,'original_summary':bvo,'permuted':base_perm,'permuted_summary':bvp})
    model.freeze();lora={'rank':args.rank,'scale':2*args.rank,'dropout':.05,'keys':LORA_KEYS};linear_to_lora_layers(model,args.layers,lora)
    trainable=dict(tree_flatten(model.trainable_parameters()));config={'fine_tune_type':'lora','num_layers':args.layers,'lora_parameters':lora};dump(out/'adapter_config.json',config)
    best=out/'best';best.mkdir(exist_ok=True);dump(best/'adapter_config.json',config);mx.save_safetensors(str(best/'adapters.safetensors'),trainable)
    optimizer=optim.AdamW(learning_rate=args.lr,weight_decay=0.)
    def loss_fn(m,row):
        z=candidate_logits(m,row);return -z[row['label']]+mx.logsumexp(z)
    vg=nn.value_and_grad(model,loss_fn);ordered=pair_order(train,args.seed);pos=0;history=[];best_score=-1.;best_step=0;started=time.perf_counter();compute=0.
    for step in range(1,total+1):
        model.train();grad_acc=None;loss_sum=0.;micro=min(args.accum,len(ordered)-pos);tick=time.perf_counter()
        for _ in range(micro):
            row=ordered[pos];pos+=1;loss,grad=vg(model,row);mx.eval(loss,grad)
            if not math.isfinite(float(loss)):raise RuntimeError('Nonfinite loss')
            grad_acc=grad if grad_acc is None else tree_map(lambda a,b:a+b,grad_acc,grad);mx.eval(grad_acc);loss_sum+=float(loss)
        grads=tree_map(lambda g:g/micro,grad_acc);grads,norm=optim.clip_grad_norm(grads,1.);lr=schedule(step,total,warmup,args.lr);optimizer.learning_rate=lr;optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state);compute+=time.perf_counter()-tick
        entry={'step':step,'loss':loss_sum/micro,'gradient_norm':float(norm),'learning_rate':lr}
        if step%16==0 or step==total:
            model.eval();vr,vs,_=evaluate(model,validation);pr,ps,_=evaluate(model,validation_shuffled);ok,reasons,score=accepted(vs,ps,bvo,bvp);entry.update(accepted=ok,reasons=reasons,original=vs,permuted=ps)
            ck=out/f'checkpoint-{step}';ck.mkdir(exist_ok=True);dump(ck/'adapter_config.json',config);weights=dict(tree_flatten(model.trainable_parameters()));mx.save_safetensors(str(ck/'adapters.safetensors'),weights)
            if ok and score>best_score:best_score=score;best_step=step;mx.save_safetensors(str(best/'adapters.safetensors'),weights)
            print(json.dumps({'step':step,'loss':entry['loss'],'accepted':ok,'accuracy':vs['by_kind']['all']['accuracy'],'pair_accuracy':vs['contrastive_family_accuracy'],'permuted_accuracy':ps['by_kind']['all']['accuracy'],'reasons':reasons}),flush=True)
        history.append(entry);dump(out/'history.json',history)
    model.eval();model.load_weights(str(best/'adapters.safetensors'),strict=False)
    print('Evaluating untouched test in two candidate orders...',flush=True);tr,ts,_=evaluate(model,test);sr,ss,_=evaluate(model,test_shuffled)
    dump(out/'test.json',{'original':tr,'original_summary':ts,'permuted':sr,'permuted_summary':ss})
    report={'contract':contract,'base_validation':{'original':bvo,'permuted':bvp},'selected_step':best_step,'accepted':best_step>0,
            'test':{'original':ts,'permuted':ss},'training_compute_seconds':compute,'wall_seconds':time.perf_counter()-started,
            'limitations':['Synthetic deterministic banking policies','Single seed','No human-reviewed banking test','Candidate permutations test position robustness, not semantic generalization']}
    dump(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
