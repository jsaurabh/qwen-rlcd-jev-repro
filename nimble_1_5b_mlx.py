"""Scale Bespoke Nimble's public recipe to quantized Qwen2.5-1.5B on MLX."""
import argparse, hashlib, json, math, random, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

ROOT=Path(__file__).resolve().parent
NIMBLE=ROOT/'reference-nimble'
sys.path.insert(0,str(NIMBLE))
from nimble.training.schema_data import as_scoring, encode_scoring
from nimble.scoring.parallel_schema import SYSTEM_PROMPT

EXPECTED={
    'train.jsonl':'beadbb9b81837f7c339e090cd210ce91f65a55f2a3ada1ce9b623765b8d6fe2e',
    'eval.jsonl':'8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896',
}
LORA_KEYS=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def raw_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def audit_raw(train_path,eval_path):
    for path in (train_path,eval_path):
        if sha(path)!=EXPECTED[path.name]:raise ValueError(f'Nimble frozen bytes differ: {path}')
    train,holdout=raw_rows(train_path),raw_rows(eval_path)
    if len(train)!=2676 or len(holdout)!=324:raise ValueError('Unexpected Nimble dataset size')
    if len({r['id'] for r in train+holdout})!=len(train)+len(holdout):raise ValueError('Duplicate record ID')
    tf={r['source_family'] for r in train};ef={r['source_family'] for r in holdout}
    if tf&ef:raise ValueError('Source family leakage')
    if any(not r['evidence_certificate']['necessity_checks_passed'] for r in train+holdout):raise ValueError('Failed evidence certificate included')
    groups=defaultdict(list)
    for r in train+holdout:groups[(r['split'],r['family'])].append(r)
    bad=[]
    for key,rows in groups.items():
        variants={r['variant'] for r in rows};targets={json.dumps(r['reference']['target'],sort_keys=True) for r in rows}
        if variants!={'base','counterfactual'} or len(targets)!=2:bad.append(key)
    if bad:raise ValueError(f'Incomplete/non-flipping contrastive pairs: {bad[:3]}')
    return train,holdout,{'train_rows':len(train),'holdout_rows':len(holdout),'train_pairs':len(train)//2,'holdout_pairs':len(holdout)//2,
        'training_source_families':len(tf),'holdout_source_families':len(ef),'source_family_overlap':False,
        'all_evidence_certificates_passed':True,'all_pairs_flip_label':True,'sha256':EXPECTED}

def encode(rows,tokenizer,max_tokens,training,shuffle_seed=None):
    out=[]
    for raw in rows:
        scoring=as_scoring(raw,training);item=encode_scoring(scoring,tokenizer,max_tokens,shuffle_seed)
        out.append({'id':raw['id'],'pair':raw['family'],'family':raw['source_family'],'domain':raw['domain'],
                    'kind':raw['input']['questions']['decision']['type'],'input_ids':item['prompt_token_ids'],
                    'candidate_ids':item['candidate_token_ids'],'label':item['target_index'],
                    'choices':list(item['code_to_choice'].values())})
    return out

def candidate_logits(model,row):
    # Candidate-only loss at the last prompt position. Avoid seq_len x vocabulary logits.
    hidden=model.model(mx.array([row['input_ids']]))[:,-1,:]
    full=model.model.embed_tokens.as_linear(hidden)
    return full[0,mx.array(row['candidate_ids'])].astype(mx.float32)

def record(row,z):
    p=np.asarray(mx.softmax(z).tolist(),dtype=float);pred=int(p.argmax());gold=row['label']
    return {'id':row['id'],'pair':row['pair'],'family':row['family'],'domain':row['domain'],'kind':row['kind'],
            'label':gold,'prediction':pred,'correct':pred==gold,'confidence':float(p[pred]),
            'reference_probability':float(p[gold]),'nll':-math.log(max(float(p[gold]),1e-15)),
            'brier':float(np.sum((p-np.eye(len(p))[gold])**2)),'probabilities':p.tolist(),
            'candidate_ids':row['candidate_ids'],'choices':row['choices']}

def summarize(records):
    by={}
    for kind in ('all','choice','noul','score'):
        rows=[r for r in records if kind=='all' or r['kind']==kind]
        if rows:by[kind]={'count':len(rows),'correct':sum(r['correct'] for r in rows),'accuracy':np.mean([r['correct'] for r in rows]).item(),'nll':np.mean([r['nll'] for r in rows]).item(),'brier':np.mean([r['brier'] for r in rows]).item()}
    pairs=defaultdict(list)
    for r in records:pairs[r['pair']].append(r['correct'])
    conf=np.array([r['confidence'] for r in records]);correct=np.array([r['correct'] for r in records]);ece=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-9))
        if mask.any():ece+=float(mask.mean()*abs(conf[mask].mean()-correct[mask].mean()))
    two=[v for v in pairs.values() if len(v)==2]
    return {'by_kind':by,'contrastive_family_accuracy':np.mean([all(v) for v in pairs.values()]).item(),
            'contrastive_pair_accuracy':np.mean([all(v) for v in two]).item() if two else None,
            'families':len(pairs),'two_member_pairs':len(two),'ece_10_bins':ece,
            'selected_position_counts':dict(Counter(r['prediction'] for r in records))}

def evaluate(model,rows):
    model.eval();start=time.perf_counter();records=[]
    for i,row in enumerate(rows,1):
        z=candidate_logits(model,row);mx.eval(z);records.append(record(row,z))
        if i%50==0:print(json.dumps({'evaluation_progress':i,'total':len(rows)}),flush=True)
    return records,summarize(records),time.perf_counter()-start

def schedule(step,total,warmup,peak):
    if step<=warmup:return peak*step/max(1,warmup)
    return peak*max(0.,(total-step)/max(1,total-warmup))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/qwen-1.5b-4bit');ap.add_argument('--out',default='runs/nimble-1.5b-v1')
    ap.add_argument('--rank',type=int,default=8);ap.add_argument('--layers',type=int,default=8);ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--accum',type=int,default=8);ap.add_argument('--max-tokens',type=int,default=1152);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--max-steps',type=int);ap.add_argument('--eval-limit',type=int);ap.add_argument('--skip-shuffled-eval',action='store_true');args=ap.parse_args()
    if min(args.rank,args.layers,args.accum,args.max_tokens)<=0 or args.lr<=0:ap.error('positive settings required')
    out=ROOT/args.out
    if (out/'report.json').exists():ap.error('completed output exists; choose a new --out')
    train_raw,eval_raw,data_audit=audit_raw(NIMBLE/'data/train.jsonl',NIMBLE/'data/eval.jsonl')
    mx.random.seed(args.seed);mx.set_cache_limit(1024*1024**2);model,tok=load(str(ROOT/args.model));model.eval()
    train=encode(train_raw,tok,args.max_tokens,True,args.seed);holdout=encode(eval_raw,tok,args.max_tokens,False)
    shuffled=encode(eval_raw,tok,args.max_tokens,False,args.seed)
    if args.eval_limit:
        holdout=holdout[:args.eval_limit];shuffled=shuffled[:args.eval_limit]
    data_audit.update(train_max_tokens=max(len(r['input_ids']) for r in train),holdout_max_tokens=max(len(r['input_ids']) for r in holdout),
                      kinds=dict(Counter(r['kind'] for r in train)),candidate_order_training=f'seeded per record, seed {args.seed}')
    total=math.ceil(len(train)/args.accum);total=min(total,args.max_steps) if args.max_steps else total;warmup=max(1,round(total*.1))
    prompt_hash=sha(NIMBLE/'nimble/scoring/parallel_schema.py');model_revision=json.loads((ROOT/args.model/'config.json').read_text()).get('_commit_hash','local')
    lora={'rank':args.rank,'scale':2*args.rank,'dropout':.05,'keys':LORA_KEYS}
    contract={'task':'schema_candidate_classification_v1','source_recipe':'Bespoke Nimble','source_commit':'f136b3f75721fda4ea961f73993cc50b08488835',
              'model':'mlx-community/Qwen2.5-1.5B-Instruct-4bit','model_path':args.model,'revision':model_revision,'system_prompt':SYSTEM_PROMPT,
              'prompt_code_sha256':prompt_hash,'max_length':args.max_tokens,'quantized_base':True,'lora_rank':args.rank,'lora_layers':args.layers,
              'lora_targets':LORA_KEYS,'seed':args.seed,'learning_rate':args.lr,'microbatch':1,'gradient_accumulation':args.accum,
              'effective_batch_size':args.accum,'warmup_steps':warmup,'max_steps':total,'lr_scheduler':'linear','weight_decay':0.,'data_audit':data_audit,
              'intentional_scale_changes':['Qwen2.5-1.5B 4-bit base','MLX training','rank/layer count reduced for 16 GiB M1 Pro','1152-token cap covers every retained record']}
    out.mkdir(parents=True,exist_ok=True);dump(out/'schema_config.json',contract);dump(out/'data_audit.json',data_audit)
    print('Evaluating frozen holdout before training...',flush=True);before,before_summary,before_seconds=evaluate(model,holdout);dump(out/'before.json',{'summary':before_summary,'rows':before})
    model.freeze();linear_to_lora_layers(model,args.layers,lora);trainable=dict(tree_flatten(model.trainable_parameters()))
    if not trainable or any('lora_' not in k for k in trainable):raise RuntimeError('Unexpected trainable parameters')
    dump(out/'adapter_config.json',{'fine_tune_type':'lora','num_layers':args.layers,'lora_parameters':lora})
    optimizer=optim.AdamW(learning_rate=args.lr,weight_decay=0.);rng=random.Random(args.seed);rng.shuffle(train)
    def loss_fn(m,row):
        z=candidate_logits(m,row);return -z[row['label']]+mx.logsumexp(z)
    vg=nn.value_and_grad(model,loss_fn);history=[];position=0;started=time.perf_counter();compute=0.
    print(json.dumps({'training_steps':total,'effective_batch':args.accum,'trainable_parameters':sum(x.size for x in trainable.values())}),flush=True)
    for step in range(1,total+1):
        model.train();grad_acc=None;loss_sum=0.;tick=time.perf_counter()
        micro=min(args.accum,len(train)-position)
        for _ in range(micro):
            row=train[position];position+=1;loss,grad=vg(model,row);mx.eval(loss,grad)
            if not math.isfinite(float(loss)):raise RuntimeError('Nonfinite loss')
            grad_acc=grad if grad_acc is None else tree_map(lambda a,b:a+b,grad_acc,grad);mx.eval(grad_acc);loss_sum+=float(loss)
        grads=tree_map(lambda g:g/micro,grad_acc);grads,norm=optim.clip_grad_norm(grads,1.);lr=schedule(step,total,warmup,args.lr);optimizer.learning_rate=lr;optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state);compute+=time.perf_counter()-tick
        entry={'step':step,'loss':loss_sum/micro,'microbatches':micro,'gradient_norm':float(norm),'learning_rate':lr};history.append(entry)
        if step%10==0 or step==total:print(json.dumps(entry),flush=True);dump(out/'history.json',history)
    training_seconds=time.perf_counter()-started;weights=dict(tree_flatten(model.trainable_parameters()));mx.save_safetensors(str(out/'adapters.safetensors'),weights)
    model.eval();print('Evaluating frozen holdout after training...',flush=True);after,after_summary,after_seconds=evaluate(model,holdout);dump(out/'after.json',{'summary':after_summary,'rows':after})
    shuffled_summary=None
    if not args.skip_shuffled_eval:
        print('Evaluating candidate-order permutation...',flush=True);permuted,shuffled_summary,shuffled_seconds=evaluate(model,shuffled);dump(out/'after_shuffled.json',{'summary':shuffled_summary,'rows':permuted})
    report={'contract':contract,'before':before_summary,'after':after_summary,'after_shuffled':shuffled_summary,
            'training':{'steps':total,'examples_seen':position,'compute_seconds':compute,'wall_seconds':training_seconds},
            'evaluation_seconds':{'before':before_seconds,'after':after_seconds},
            'limitations':['Synthetic model-checked labels, not human-reviewed','Single seeded fit','Qwen2.5-1.5B hardware-scaled recipe differs from published Nimble 9B fit','No temperature calibration','Frozen holdout used only for final before/after reporting']}
    dump(out/'report.json',report);print(json.dumps({'report':str(out/'report.json'),'before':before_summary,'after':after_summary,'after_shuffled':shuffled_summary},indent=2),flush=True)

if __name__=='__main__':main()
