"""Collect pinned source training inputs privately; no source code execution."""
from pathlib import Path
import json,urllib.request,hashlib,time
from huggingface_hub import HfApi
from datasets import load_dataset
R=Path('/content/data-v3/work/sources');R.mkdir(exist_ok=True)
manifest={'seed':20261009,'sources':{},'status':'collecting','redistribution':'private inputs only; check each source terms before training/publication'}
def save():
 (R/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'status':manifest['status'],'sources':{k:v.get('status') for k,v in manifest['sources'].items()}}),flush=True)
def fetch(url,p):
 with urllib.request.urlopen(url,timeout=120) as src,p.open('wb') as dst:
  while b:=src.read(8*1024*1024):dst.write(b)
 return hashlib.sha256(p.read_bytes()).hexdigest()
for name,repo,rev,files in [
 ('acos','NUSTM/ACOS','45d179a3dcc6a3dedd848d81b16f2552454805fe',['data/Laptop-ACOS/laptop_quad_train.tsv','data/Restaurant-ACOS/rest16_quad_train.tsv','README.md']),
 ('esci','amazon-science/esci-data','7916cdf6ab75a462e77f20ab40428a10923998d5',['shopping_queries_dataset/shopping_queries_dataset_examples.parquet','shopping_queries_dataset/shopping_queries_dataset_products.parquet','LICENSE','README.md'])]:
 info={'repo':repo,'revision':rev,'status':'downloading','files':{},'license_review':'pending'};manifest['sources'][name]=info;save();d=R/name;d.mkdir(exist_ok=True)
 try:
  for f in files:
   url=f'https://raw.githubusercontent.com/{repo}/{rev}/{f}';p=d/Path(f).name
   digest=fetch(url,p)
   if p.stat().st_size<1000 and p.read_bytes().startswith(b'version https://git-lfs.github.com'):
    digest=fetch(f'https://media.githubusercontent.com/media/{repo}/{rev}/{f}',p)
   info['files'][f]={'local':p.name,'sha256':digest,'bytes':p.stat().st_size}
  if name=='esci':
   import pyarrow.parquet as pq
   info['schemas']={f.name:pq.read_schema(f).names for f in d.glob('*.parquet')}
  info['status']='downloaded'
 except Exception as e:info['status']='failed';info['error_type']=type(e).__name__
 save()
for name,repo,limit in [('numina','AI-MO/NuminaMath-CoT',20000),('tulu','allenai/tulu-3-sft-mixture',10000),('xlam','Salesforce/xlam-function-calling-60k',10000)]:
 info={'repo':repo,'status':'resolving','license_review':'pending'};manifest['sources'][name]=info;save()
 try:
  meta=HfApi().dataset_info(repo);info['revision']=meta.sha;info['declared_license']=meta.card_data.get('license') if meta.card_data else None
  ds=load_dataset(repo,split='train',revision=meta.sha,streaming=True).shuffle(seed=20261009,buffer_size=5000)
  p=R/(name+'-train.jsonl');count=0
  with p.open('w') as f:
   for row in ds:
    if not count:info['columns']=list(row)
    f.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
    if count>=limit:break
  info.update(status='downloaded',rows=count,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e:info.update(status='failed',error_type=type(e).__name__)
 save()
manifest['status']='source_collection_complete';save()
(R/'COMPLETE.json').write_text(json.dumps({'time':time.time(),'status':manifest['status']}))
