"""Cache a frozen reference only for the exact planned decision inputs."""
from pathlib import Path
import hashlib,json,os
import torch

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()

def artifact_hash(root):
 root=Path(root);paths=[root/'readout.safetensors',root/'decision_config.json',*sorted((root/'adapter').glob('*'))]
 return hashlib.sha256(json.dumps({str(p.relative_to(root)):sha(p) for p in paths if p.is_file()},sort_keys=True).encode()).hexdigest()

def reference_cache(model,rows,plan,path,identity,batch_size,allow_create=True):
 path=Path(path);meta_path=path.with_suffix('.json')
 if path.exists() or meta_path.exists():
  meta=json.loads(meta_path.read_text())
  if meta['identity']!=identity or meta['sha256']!=sha(path):raise ValueError('Reference cache identity/hash mismatch')
  values=torch.load(path,map_location='cpu',weights_only=True)
 else:
  if not allow_create:raise ValueError('Resume needs the original frozen reference cache')
  model.eval();chunks=[]
  with torch.inference_mode():
   for start in range(0,len(plan),batch_size):
    part=[rows[i] for i in plan[start:start+batch_size]];batch=model.prepare(part,max_length=identity['max_length'])
    if batch.inputs['attention_mask'].sum(1).tolist()!=[r['input_token_count'] for r in part]:raise ValueError('Reference token mismatch')
    chunks.append(model(batch).float().log_softmax(-1).cpu())
    if start%128==0:print(json.dumps({'phase':'reference','rows':min(start+batch_size,len(plan)),'total':len(plan)}),flush=True)
   values=torch.cat(chunks)
  path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(values,tmp);os.replace(tmp,path)
  meta={'identity':identity,'sha256':sha(path),'rows':len(plan)};tmp_meta=meta_path.with_suffix('.tmp.json');tmp_meta.write_text(json.dumps(meta,indent=2));os.replace(tmp_meta,meta_path)
  # Reload as normal tensors, not inference-mode tensors used by autograd.
  values=torch.load(path,map_location='cpu',weights_only=True)
 if values.shape!=(len(plan),255):raise ValueError('Reference shape mismatch')
 model.train();return values,meta
