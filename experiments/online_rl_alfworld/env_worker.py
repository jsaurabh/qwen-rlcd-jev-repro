"""ALFWorld JSON-lines server. Exposes observations/legal actions, never oracle plans."""
import contextlib,json,sys,random,hashlib
from pathlib import Path
import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
ENVS={}
def visible(s):
 return {'observation':s.feedback,'commands':list(s.admissible_commands),'won':bool(s.won)}
def dispatch(q):
 op=q['op']
 if op=='catalog':
  root=Path(q['root']);result={}
  for split in ['train','valid_seen','valid_unseen']:
   found=[]
   for p in sorted(root.glob(f'**/{split}/**/game.tw-pddl')):
    if 'movable' in str(p) or 'Sliced' in str(p):continue
    data=json.loads(p.read_text())
    if not data.get('solvable'):continue
    meta=json.loads((p.parent/'traj_data.json').read_text())
    found.append({'path':str(p),'id':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'task_type':meta['task_type']})
   result[split]=found
  return result
 if op=='reset':
  for env in ENVS.values():env.close()
  ENVS.clear();states=[]
  for i,path in enumerate(q['paths']):
   env=textworld.start(path,request_infos=textworld.EnvInfos(won=True,admissible_commands=True),wrappers=[AlfredDemangler])
   env.seed(q.get('seed',0));ENVS[i]=env;states.append(visible(env.reset()))
  return states
 if op=='step':
  out=[]
  for item in q['actions']:
   s,_,done=ENVS[item['index']].step(item['command'])
   out.append({'index':item['index'],**visible(s),'done':bool(done)})
  return out
 if op=='close':
  for e in ENVS.values():e.close()
  ENVS.clear();return {'closed':True}
 raise ValueError(op)
for line in sys.stdin:
 try:
  with contextlib.redirect_stdout(sys.stderr):result=dispatch(json.loads(line))
  print(json.dumps({'result':result}),flush=True)
 except BaseException as e:
  import traceback;traceback.print_exc(file=sys.stderr)
  print(json.dumps({'error':type(e).__name__+': '+str(e)}),flush=True)
