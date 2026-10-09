"""Validate decisions, freeze option order and tokenize without loading model weights."""
from pathlib import Path
from collections import Counter,defaultdict
import os,json,random,hashlib,ast,time
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
R=Path('/content/data-v3/work');I=R/'screened';O=R/'ready';O.mkdir(exist_ok=True)
repo='jsaurabh/qwen-decision-27b-noncausal-lora';rev='b2eefb929a883136b3bd56ce70999fdac656ee8a'
cfg=json.loads(Path(hf_hub_download(repo,'decision_config.json',revision=rev)).read_text())
tok=AutoTokenizer.from_pretrained(repo,subfolder='processor',revision=rev)
tok.chat_template=Path(hf_hub_download(repo,'processor/chat_template.jinja',revision=rev)).read_text()
code=Path(hf_hub_download(repo,'autojev/model.py',revision=rev)).read_text();tree=ast.parse(code)
funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['describe','options','decision_messages']]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*funcs],type_ignores=[])
env={'json':json,'MAX_OPTIONS':255};exec(compile(ast.fix_missing_locations(module),'frozen-render-functions','exec'),env)
render=env['decision_messages'];options=env['options'];codes=cfg['codes']
assert [tok.encode(c,add_special_tokens=False)[0] for c in codes]==cfg['token_ids']
def status(phase,**kw):
 (R/'token-status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)));print(json.dumps(dict(phase=phase,**kw)),flush=True)
def freeze(row):
 q=row['question'];target=row['target'];keys,_=options(q)
 assert 1<=len(keys)<=255
 if q['type']=='choice':
  if isinstance(target,list):assert len(target)==len(keys) and all(float(v)>=0 for v in target) and abs(sum(target)-1)<1e-4
  else:assert str(target) in keys
  order=list(q['criteria'].items());random.Random('20261009:'+row['id']).shuffle(order)
  if isinstance(target,list):dist=dict(zip(keys,target));row['target']=[dist[k] for k,_ in order]
  q['criteria']=dict(order)
 elif q['type']=='noul':
  assert isinstance(target,(bool,int,float,list))
  if isinstance(target,list):assert len(target)==2 and abs(sum(target)-1)<1e-4
  else:assert 0<=float(target)<=1
 else:raise ValueError('Unexpected training primitive')
 return row
report={'tokenizer_repo':repo,'revision':rev,'max_length':512,'filter_limit':480,'option_order':'Frozen per-example permutation seed20261009+id; trainer must not reshuffle labels afterward.','splits':{}}
for name in ['control-train','targeted-train','dev','temperature']:
 count=Counter();suite=Counter();kept=[];token_total=0
 with (I/f'{name}.jsonl').open() as stream: rows=[json.loads(l) for l in stream if l.strip()]
 for start in range(0,len(rows),256):
  part=[freeze(r) for r in rows[start:start+256]]
  texts=[tok.apply_chat_template(render(r,codes),tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in part]
  ids=tok(texts,add_special_tokens=False)['input_ids']
  for r,encoded,rendered in zip(part,ids,texts):
   count['input']+=1
   if len(encoded)>480:count['over_length']+=1;continue
   r['input_token_count']=len(encoded);r['render_sha256']=hashlib.sha256(rendered.encode()).hexdigest()
   kept.append(r);suite[r.get('suite','unknown')]+=1;token_total+=len(encoded)
  if start%8192==0:status('tokenizing',split=name,processed=start,kept=len(kept))
 if name in ['dev','temperature']:
  groups=defaultdict(list)
  for r in kept:groups[r.get('suite','unknown')].append(r)
  for k,rs in groups.items():random.Random('eval:'+name+k).shuffle(rs)
  selected=[]
  while len(selected)<min(768,len(kept)):
   for k in sorted(groups):
    if groups[k] and len(selected)<768:selected.append(groups[k].pop())
  kept=selected;suite=Counter(r.get('suite','unknown') for r in kept);token_total=sum(r['input_token_count'] for r in kept)
 path=O/f'{name}.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept))
 report['splits'][name]={'counts':dict(count),'kept':len(kept),'by_suite':dict(suite),'tokens':token_total,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(O/'token-manifest.json').write_text(json.dumps(report,indent=2));(O/'COMPLETE.json').write_text(json.dumps({'time':time.time()}));status('tokenization_complete',report=report)
