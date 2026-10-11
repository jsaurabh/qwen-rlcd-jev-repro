"""Online multi-step ALFWorld decision-policy RL; no answer labels or SFT term."""
from pathlib import Path
import argparse,copy,json,os,random,subprocess,sys,time,hashlib,shutil
import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors.torch import load_file,save_file
from autojev.model import DecisionModel
from autojev_lora_train import enable_noncausal_full_attention,evaluate,infer
from autojev.evaluate import read_rows,metrics,evaluate_logits
from policy_objective import leave_one_out,trajectory_return,policy_loss,matching_rollout_batches

class EnvClient:
 def __init__(self,python,script):
  self.proc=subprocess.Popen([python,'-u',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
 def call(self,**request):
  self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush()
  line=self.proc.stdout.readline()
  if not line:raise RuntimeError('Environment server exited')
  response=json.loads(line)
  if 'error' in response:raise RuntimeError(response['error'])
  return response['result']
 def close(self):
  self.call(op='close');self.proc.stdin.close();self.proc.wait(timeout=20)

def emit(event,**fields):print(json.dumps({'event':event,'time':time.time(),**fields}),flush=True)
def append(path,row):
 with path.open('a') as f:f.write(json.dumps(row)+'\n')

def decision_row(initial,history,commands,steps_left,model,max_length,rng):
 commands=list(commands);rng.shuffle(commands)
 if not commands or len(commands)>255:raise ValueError('Unsupported legal action count')
 history=list(history)
 while True:
  row={'state':{'initial_observation':initial,'recent_interactions':history,'steps_remaining':steps_left},'question':{'type':'choice','instructions':'Select the next legal action to complete the stated task. Use observations and previous actions to find objects and operate appliances. Avoid repeating unproductive actions.','criteria':{c:None for c in commands}}}
  # Explicitly drop oldest complete interactions if necessary. Goal and current options are never truncated.
  try:model.prepare([row],max_length=max_length);return row,commands
  except ValueError as e:
   if 'token limit' not in str(e) or len(history)<=1:raise
   history.pop(0)

@torch.no_grad()
def scores(model,rows,micro,max_length,reference=False):
 if reference:
  model.backbone.set_adapter('reference')
  for n,p in model.backbone.named_parameters():
   if '.reference.' in n:p.requires_grad_(False)
 else:model.backbone.set_adapter('default')
 model.eval();all_logits=[]
 try:
  for start in range(0,len(rows),micro):
   batch=model.prepare(rows[start:start+micro],max_length=max_length)
   if reference:
    hidden=model.backbone(**batch.inputs,use_cache=False).last_hidden_state[:,-1]
    logits=F.linear(hidden,model.frozen_readout).float()
   else:logits=model(batch)
   all_logits.extend(logits[i,:n].cpu() for i,n in enumerate(batch.counts))
 finally:
  model.backbone.set_adapter('default')
 return all_logits

def rollout(model,env,tasks,args,rng,sample,with_reference):
 states=env.call(op='reset',paths=[t['path'] for t in tasks],seed=args.seed)
 initial=[s['observation'] for s in states];histories=[[] for s in states]
 traj=[{'task':t['id'],'task_type':t['task_type'],'transitions':[],'won':False,'done':False} for t in tasks]
 for step in range(args.horizon):
  active=[i for i,t in enumerate(traj) if not t['done']]
  if not active:break
  pairs=[decision_row(initial[i],histories[i][-12:],states[i]['commands'],args.horizon-step,model,args.max_length,rng) for i in active]
  rows=[p[0] for p in pairs];logits=scores(model,rows,args.micro_batch,args.max_length)
  ref=scores(model,rows,args.micro_batch,args.max_length,True) if with_reference else None
  acts=[]
  for j,i in enumerate(active):
   lp=(logits[j]/args.temperature).log_softmax(-1)
   action=int(torch.multinomial(lp.exp(),1)) if sample else int(lp.argmax())
   rlp=(ref[j]/args.temperature).log_softmax(-1)[action].item() if ref is not None else lp[action].item()
   command=pairs[j][1][action]
   traj[i]['transitions'].append({'row':rows[j],'action':action,'old_logp':lp[action].item(),'reference_logp':rlp,'command':command})
   acts.append({'index':i,'command':command})
  next_states=env.call(op='step',actions=acts)
  for state in next_states:
   i=state['index'];states[i]=state
   histories[i].append({'action':traj[i]['transitions'][-1]['command'],'observation':state['observation']})
   traj[i]['transitions'][-1]['observation']=state['observation']
   traj[i]['won']=state['won'];traj[i]['done']=state['done'] or state['won']
  if step%10==0:emit('rollout_progress',step=step+1,active=len(active),won=sum(t['won'] for t in traj))
 for t in traj:
  t['steps']=len(t['transitions']);t['sampled_log_ratio']=sum(x['old_logp']-x['reference_logp'] for x in t['transitions'])
  t['return']=trajectory_return(t['won'],t['steps'],t['sampled_log_ratio'],args.beta,args.step_cost)
 return traj

def summarize(trajectories):
 return {'episodes':len(trajectories),'success_rate':sum(t['won'] for t in trajectories)/len(trajectories),'mean_steps':sum(t['steps'] for t in trajectories)/len(trajectories),'mean_return':sum(t['return'] for t in trajectories)/len(trajectories)}

def evaluate_env(model,env,tasks,args,seed):
 rng=random.Random(seed);all_t=[]
 for start in range(0,len(tasks),args.eval_batch):all_t.extend(rollout(model,env,tasks[start:start+args.eval_batch],args,rng,False,False))
 by_type={}
 for typ in sorted({t['task_type'] for t in all_t}):by_type[typ]=summarize([t for t in all_t if t['task_type']==typ])
 return {'overall':summarize(all_t),'by_type':by_type},all_t

def checkpoint(model,optimizer,out,step,rng,config,mailbox,name=None):
 dst=out/(name or f'checkpoint-{step:05d}');tmp=out/('.partial-'+dst.name)
 if dst.exists() or tmp.exists():raise FileExistsError(dst)
 tmp.mkdir()
 model.backbone.set_adapter('default')
 model.backbone.save_pretrained(tmp/'adapter',selected_adapters=['default'])
 save_file({'weight':model.readout.weight.detach().cpu().contiguous()},str(tmp/'readout.safetensors'))
 model.processor.save_pretrained(tmp/'processor')
 cfg={**config,'step':step,'temperature':1.0,'codes':model.codes,'token_ids':model.token_ids}
 (tmp/'decision_config.json').write_text(json.dumps(cfg,indent=2))
 torch.save({'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':rng.getstate(),'step':step},tmp/'training_state.pt')
 for name in ['run_config.json','training.jsonl','evaluations.jsonl','split_manifest.json','baseline.json','final.json','rollouts.jsonl','baseline-valid_seen-trajectories.jsonl','baseline-valid_unseen-trajectories.jsonl','final-valid_seen-trajectories.jsonl','final-valid_unseen-trajectories.jsonl']:
  if (out/name).exists():shutil.copyfile(out/name,tmp/name)
 (tmp/'COMPLETE.json').write_text(json.dumps({'step':step}));tmp.rename(dst)
 receipt=mailbox.upload_checkpoint(dst);emit('checkpoint_verified',step=step,**receipt)


def main():
 p=argparse.ArgumentParser()
 for n in ['initial','out','data','env-python','env-worker','checkpoint-spool','retention-data']:p.add_argument('--'+n,required=True)
 p.add_argument('--updates',type=int,default=64);p.add_argument('--tasks-per-update',type=int,default=2);p.add_argument('--group-size',type=int,default=4)
 p.add_argument('--horizon',type=int,default=50);p.add_argument('--max-length',type=int,default=1536);p.add_argument('--micro-batch',type=int,default=4)
 p.add_argument('--eval-batch',type=int,default=8);p.add_argument('--eval-tasks',type=int,default=24);p.add_argument('--retention-rows',type=int,default=768)
 p.add_argument('--temperature',type=float,default=1.0);p.add_argument('--beta',type=float,default=.005);p.add_argument('--step-cost',type=float,default=.002)
 p.add_argument('--lr',type=float,default=2e-6);p.add_argument('--seed',type=int,default=20261010);p.add_argument('--max-training-hours',type=float,default=6)
 p.add_argument('--resume');p.add_argument('--smoke-only',action='store_true');args=p.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=True,parents=True);rng=random.Random(args.seed);torch.manual_seed(args.seed)
 from checkpoint_mailbox import CheckpointMailbox
 mailbox=CheckpointMailbox(Path(args.checkpoint_spool))
 env=EnvClient(args.env_python,args.env_worker)
 catalog=env.call(op='catalog',root=args.data)
 assert all(catalog[k] for k in ['train','valid_seen','valid_unseen'])
 # Keep the official evaluation worlds entirely out of rollouts and optimization.
 train=sorted(catalog['train'],key=lambda t:t['id']);evals={}
 for split in ['valid_seen','valid_unseen']:
  ordered=[]
  for typ in sorted({t['task_type'] for t in catalog[split]}):
   items=[t for t in catalog[split] if t['task_type']==typ];random.Random(args.seed).shuffle(items);ordered.append(items)
  import itertools
  evals[split]=[t for group in itertools.zip_longest(*ordered) for t in group if t is not None][:args.eval_tasks]
 manifest={'train':train,**evals};(out/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
 evalhash={t['sha256'] for tasks in evals.values() for t in tasks}
 assert not evalhash.intersection(t['sha256'] for t in train)
 initial=Path(args.initial);cfg=json.loads((initial/'decision_config.json').read_text())
 emit('loading_model',train_tasks=len(train),eval_tasks={k:len(v) for k,v in evals.items()})
 model=DecisionModel(train=True,base_model=cfg['base_model'],revision=cfg['revision'],gradient_checkpointing=True,cpu_threads=8)
 enable_noncausal_full_attention(model.backbone.language_model)
 model.backbone=PeftModel.from_pretrained(model.backbone,Path(args.resume or initial)/'adapter',is_trainable=True)
 model.backbone.load_adapter(initial/'adapter',adapter_name='reference',is_trainable=False)
 model.readout.load_state_dict(load_file(str(Path(args.resume or initial)/'readout.safetensors')))
 model.register_buffer('frozen_readout',load_file(str(initial/'readout.safetensors'))['weight'].to(model.device_name))
 model.backbone.set_adapter('default');model.readout.requires_grad_(True)
 # Identical rollout/update distributions: disable stochastic dropout, not autograd.
 for module in model.modules():
  if isinstance(module,torch.nn.Dropout):module.p=0.0
 params=[p for p in model.parameters() if p.requires_grad]
 assert all(not p.requires_grad for n,p in model.backbone.named_parameters() if '.reference.' in n)
 optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.0)
 config={**vars(args),'format_version':1,'architecture':'autojev-readout+lora','base_model':cfg['base_model'],'revision':cfg['revision'],'attention_mode':'noncausal_full_attention','algorithm':'on-policy trajectory REINFORCE leave-one-out; one optimizer update per rollout batch; no CE','lora_rank':16,'lora_alpha':32,'lora_dropout':0.0,'reference_initialization':'completed full-epoch SFT','trainable_parameters':sum(p.numel() for p in params),'alfworld_revision':'aaba6870f86c5be6a08a491f32a50b906227bc3e'}
 (out/'run_config.json').write_text(json.dumps(config,indent=2))
 dev=read_rows(Path(args.retention_data)/'dev.jsonl')[:args.retention_rows];cal=read_rows(Path(args.retention_data)/'temperature.jsonl')[:args.retention_rows]
 start=0
 if args.resume:
  state=torch.load(Path(args.resume)/'training_state.pt',weights_only=False,map_location='cpu');optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);rng.setstate(state['python_rng']);start=state['step']
  saved=json.loads((Path(args.resume)/'decision_config.json').read_text())
  for key in ['seed','lr','beta','horizon','max_length','group_size','tasks_per_update','temperature']:
   assert saved[key]==config[key],f'Resume mismatch {key}'
 if not args.resume:
  # Check reference switching and rollout/update equivalence before expensive evaluation.
  probe_tasks=train[:args.micro_batch]
  probe_states=env.call(op='reset',paths=[t['path'] for t in probe_tasks],seed=args.seed)
  probe_rows=[decision_row(s['observation'],[],s['commands'],args.horizon,model,args.max_length,random.Random(args.seed))[0] for s in probe_states]
  policy=scores(model,probe_rows,args.micro_batch,args.max_length)
  reference=scores(model,probe_rows,args.micro_batch,args.max_length,True)
  for a,b in zip(policy,reference):torch.testing.assert_close(a,b,rtol=0,atol=0)
  model.train();probe=model.prepare(probe_rows,max_length=args.max_length);lp=model(probe).log_softmax(-1)
  delta=max(float((lp[i,:len(a)].detach().cpu()-a.log_softmax(-1)).abs().max()) for i,a in enumerate(policy))
  assert delta<.15,('preflight policy mismatch',delta)
  loss=-lp[:,0].mean();loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(params,1.0)
  assert torch.isfinite(norm) and norm>0
  assert all(p.grad is None for n,p in model.backbone.named_parameters() if '.reference.' in n)
  optimizer.zero_grad(set_to_none=True)
  emit('gpu_preflight_passed',policy_delta=delta,gradient_norm=float(norm),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
  baseline={}
  for split,tasks in evals.items():
   baseline[split],trajectories=evaluate_env(model,env,tasks,args,args.seed+100)
   for t in trajectories:append(out/f'baseline-{split}-trajectories.jsonl',t)
   emit('baseline_environment',split=split,**baseline[split]['overall'])
  temp,retention=evaluate(model,dev,cal,args.micro_batch,512);baseline['retention']={'temperature':temp,'metrics':retention}
  (out/'baseline.json').write_text(json.dumps(baseline,indent=2));checkpoint(model,optimizer,out,0,rng,config,mailbox)
 else:baseline=json.loads((out/'baseline.json').read_text())
 training_started=time.monotonic();successes=0;mixed=0;stop_reason='update_budget';last_step=start
 for update in range(start+1,args.updates+1):
  if time.monotonic()-training_started>args.max_training_hours*3600:stop_reason='wall_clock_budget';break
  tasks=rng.sample(train,args.tasks_per_update);tasks=[t for t in tasks for _ in range(args.group_size)]
  t0=time.monotonic();trajectories=rollout(model,env,tasks,args,rng,True,True)
  advantages=leave_one_out([t['return'] for t in trajectories],args.group_size)
  successes+=sum(t['won'] for t in trajectories)
  for j in range(0,len(trajectories),args.group_size):
   wins=[t['won'] for t in trajectories[j:j+args.group_size]];mixed+=int(any(wins) and not all(wins))
  transitions=[(transition,advantages[i]) for i,t in enumerate(trajectories) for transition in t['transitions']]
  model.backbone.set_adapter('default');model.train();optimizer.zero_grad(set_to_none=True)
  loss_total=0.;max_logp_delta=0.
  for part in matching_rollout_batches(trajectories,advantages,args.micro_batch):
   batch=model.prepare([x[0]['row'] for x in part],max_length=args.max_length)
   logp=(model(batch)/args.temperature).log_softmax(-1)
   action=torch.tensor([x[0]['action'] for x in part],device=logp.device);chosen=logp.gather(1,action[:,None]).squeeze(1)
   old=torch.tensor([x[0]['old_logp'] for x in part],device=logp.device)
   delta=float((chosen.detach()-old).abs().max());max_logp_delta=max(max_logp_delta,delta)
   if delta>.15:raise RuntimeError(f'Rollout/update policy mismatch {delta}; no optimizer step taken')
   adv=torch.tensor([x[1] for x in part],device=logp.device)
   loss=policy_loss(chosen,adv,len(trajectories))
   if not torch.isfinite(loss):raise ValueError('Nonfinite policy gradient')
   loss.backward();loss_total+=float(loss.detach())
  norm=torch.nn.utils.clip_grad_norm_(params,1.0)
  if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
  assert all(p.grad is None for n,p in model.backbone.named_parameters() if '.reference.' in n)
  optimizer.step();last_step=update
  entry={'update':update,**summarize(trajectories),'loss':loss_total,'gradient_norm':float(norm),'max_logp_delta':max_logp_delta,'transitions':len(transitions),'wall_seconds':time.monotonic()-t0,'total_successes':successes,'mixed_success_groups':mixed,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
  append(out/'training.jsonl',entry);emit('update',**entry)
  for t in trajectories:append(out/'rollouts.jsonl',{'update':update,**t})
  if update%4==0 or update==args.updates:checkpoint(model,optimizer,out,update,rng,config,mailbox)
  if update==8 and (successes<4 or mixed<2):stop_reason='insufficient_terminal_reward_signal';break
  if args.smoke_only:stop_reason='smoke_only';break
 final={'updates':last_step,'stop_reason':stop_reason,'training_wall_seconds':time.monotonic()-training_started}
 for split,tasks in evals.items():
  final[split],trajectories=evaluate_env(model,env,tasks,args,args.seed+100)
  for t in trajectories:append(out/f'final-{split}-trajectories.jsonl',t)
  emit('final_environment',split=split,**final[split]['overall'])
 temp,retention=evaluate(model,dev,cal,args.micro_batch,512);final['retention']={'temperature':temp,'metrics':retention}
 (out/'final.json').write_text(json.dumps(final,indent=2))
 # Distinct final artifact includes the final report and completed trajectory log.
 checkpoint(model,optimizer,out,last_step,rng,{**config,'actual_updates':last_step,'artifact_role':'final'},mailbox,name='selected')
 env.close();emit('complete',**{k:v for k,v in final.items() if k not in ['valid_seen','valid_unseen','retention']})
if __name__=='__main__':main()
