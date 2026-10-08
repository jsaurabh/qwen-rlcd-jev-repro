"""Train Qwen3.5-9B on frozen Nimble data plus the gap curriculum."""
import argparse, hashlib, json, math, random, time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from nimble.training.model_loading import load_base
from nimble.training.schema_data import as_scoring, encode_scoring, runtime_record
from nimble.training.schema_train import CandidateCollator, CandidateTrainer, evaluate

MODEL="Qwen/Qwen3.5-9B"
REV="c202236235762e1c871ad0ccb60c8ee5ba337b9a"

def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode(rows,tok,maxlen,training,seed=None):
    out=[]
    for r in rows:
        s=as_scoring(r,training); e=encode_scoring(s,tok,maxlen,seed)
        out.append(runtime_record(e,r))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--nimble',type=Path,default=Path('/content/nimble'))
    ap.add_argument('--gap',type=Path,default=Path('/content/jev_gap_curriculum_v1'))
    ap.add_argument('--out',type=Path,default=Path('/content/nimble-gap-9b-v1'))
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--seed',type=int,default=17); ap.add_argument('--max-length',type=int,default=2048)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=bool(a.resume)); set_seed(a.seed)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV); tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    original=read(a.nimble/'data/train.jsonl'); gap_train=read(a.gap/'train.jsonl')
    original_eval=read(a.nimble/'data/eval.jsonl'); gap_eval=read(a.gap/'eval.jsonl')
    if {r['family'] for r in gap_train}&{r['family'] for r in gap_eval}: raise ValueError('gap pair leakage')
    if len({r['id'] for r in original+gap_train}) != len(original)+len(gap_train): raise ValueError('duplicate IDs')
    train=encode(original+gap_train,tok,a.max_length,True,a.seed)
    validation=encode(original_eval,tok,a.max_length,False)
    gap_validation=encode(gap_eval,tok,a.max_length,False)
    rng=random.Random(a.seed); rng.shuffle(train)
    batch,accum=2,4; steps=math.ceil(len(train)/(batch*accum)); warmup=round(steps*.1)
    audit={'original_train_rows':len(original),'gap_train_rows':len(gap_train),'combined_train_rows':len(train),
           'original_eval_rows':len(validation),'gap_eval_rows':len(gap_validation),'gap_pair_leakage':False,
           'original_train_sha256':sha(a.nimble/'data/train.jsonl'),'gap_train_sha256':sha(a.gap/'train.jsonl'),
           'gap_eval_sha256':sha(a.gap/'eval.jsonl'),'seed':a.seed,'max_length':a.max_length,
           'optimizer_steps':steps,'warmup_steps':warmup}
    (a.out/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    base=load_base(MODEL,REV)
    targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and '.language_model.' in n]
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    model.print_trainable_parameters(); collator=CandidateCollator(tok.pad_token_id)
    if a.resume:
        before_original={'summary':{'all':{'count':324,'correct':215,'accuracy':0.6635802469135802}}}
        before_gap={'summary':{'all':{'count':480,'correct':246,'accuracy':0.5125}}}
    else:
        before_original=evaluate(model,validation,collator,a.out/'before_original.json')
        before_gap=evaluate(model,gap_validation,collator,a.out/'before_gap.json')
    contract={'task':'schema_candidate_classification_v1','experiment':'nimble-plus-gap-curriculum-v1','model':MODEL,'revision':REV,
              'seed':a.seed,'max_length':a.max_length,'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
              'learning_rate':5e-5,'batch_size':batch,'gradient_accumulation':accum,'warmup_steps':warmup,
              'max_steps':steps,'weight_decay':0.0,'data_audit':audit,'target_modules':targets,
              'versions':{p:version(p) for p in ('torch','transformers','peft','accelerate')},'gpu':torch.cuda.get_device_name()}
    (a.out/'schema_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    args=TrainingArguments(output_dir=str(a.out),max_steps=steps,learning_rate=5e-5,per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,warmup_steps=warmup,lr_scheduler_type='linear',weight_decay=0.,bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='no',
        save_strategy='steps',save_steps=100,save_total_limit=2,logging_steps=10,report_to='none',remove_unused_columns=False,
        label_names=['labels'],seed=a.seed,data_seed=a.seed,dataloader_num_workers=0,max_grad_norm=1.,optim='adamw_torch',logging_nan_inf_filter=False)
    trainer=CandidateTrainer(model=model,args=args,train_dataset=train,data_collator=collator,processing_class=tok)
    torch.cuda.reset_peak_memory_stats(); start=time.monotonic(); result=trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None); elapsed=time.monotonic()-start
    after_original=evaluate(model,validation,collator,a.out/'after_original.json')
    after_gap=evaluate(model,gap_validation,collator,a.out/'after_gap.json')
    trainer.save_model(str(a.out)); tok.save_pretrained(a.out); trainer.save_state()
    report={'before_original':before_original['summary'],'after_original':after_original['summary'],
            'before_gap':before_gap['summary'],'after_gap':after_gap['summary'],'training':result.metrics,
            'wall_seconds':elapsed,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
    (a.out/'run_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
