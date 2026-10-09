from pathlib import Path
import collections,hashlib,json,random,tarfile,sys
root=Path('/content/attention-ablation')
sys.path.insert(0,str(root/'release/source/src'))
from transformers import AutoProcessor
from autojev.model import decision_messages
import itertools,string
out=root/'data'; out.mkdir(exist_ok=True)
archive=Path('/content/autojev-data-v2.tar.gz')
with tarfile.open(archive) as tar:
 folds={n:[json.loads(x) for x in tar.extractfile(n+'.jsonl') if x.strip()] for n in ['train','dev','temperature']}
# Recover a fixed, stratified 0.5-scale text subset from the archived 0.75-scale corpus.
# This is a new paired experiment, not a claim of byte-identical historical data.
groups=collections.defaultdict(list)
for row in folds['train']:
 if not row.get('images'): groups[row['suite']].append(row)
train=[]
for key,rows in sorted(groups.items()):
 rows.sort(key=lambda r:hashlib.sha256(('20261008:'+r['id']).encode()).hexdigest())
 train.extend(rows[:len(rows)*2//3])
raw_gap=[json.loads(x) for x in Path('/content/gap-train.jsonl').read_text().splitlines() if x.strip()]
for row in raw_gap:
 train.append({'id':'gap:'+row['id'],'suite':'jev-gap-curriculum-v1','family':row['family'],
 'state':row['input']['state'],'question':row['input']['questions']['decision'],
 'target':row['reference']['target'],'label':row['reference']['target'],
 'source':{'dataset':'jev-gap-curriculum-v1','split':'train','license':'MIT'}})
folds['train']=train
processor=AutoProcessor.from_pretrained('Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
candidates=list(string.ascii_uppercase)+[''.join(p) for p in itertools.product(string.ascii_uppercase,repeat=2)]
codes=[c for c in candidates if len(processor.tokenizer.encode(c,add_special_tokens=False))==1][:255]
report={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'gap_sha256':hashlib.sha256(Path('/content/gap-train.jsonl').read_bytes()).hexdigest(),
 'selection':'New deterministic 2/3 subset per suite of archived 0.75-scale text corpus; plus gap training rows. Not byte-identical to historical full-v4.',
 'max_length':512,'filter_length':480,'seed':20261008,'splits':{}}
for name,rows in folds.items():
 kept=[]
 for start in range(0,len(rows),256):
  part=rows[start:start+256]
  texts=[processor.apply_chat_template(decision_messages(r,codes),tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in part]
  counts=[len(x) for x in processor.tokenizer(texts,add_special_tokens=False)['input_ids']]
  for row,count in zip(part,counts):
   # Reserve 32 tokens for changes caused by option order, verified again at training time.
   if count<=480: kept.append(row)
  if start%8192==0: print(json.dumps({'split':name,'processed':min(start+256,len(rows))}),flush=True)
 path=out/(name+'.jsonl')
 path.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in kept))
 report['splits'][name]={'original':len(rows),'kept':len(kept),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
# Namespaced family overlap and IDs are checked before model training.
def families(rows): return {(r['source']['dataset'],r['family']) for r in rows}
sets={n:families([json.loads(x) for x in (out/(n+'.jsonl')).read_text().splitlines()]) for n in folds}
for a,b in [('train','dev'),('train','temperature'),('dev','temperature')]:
 overlap=sets[a]&sets[b]
 if overlap: raise ValueError(f'Family leakage between {a} and {b}: {len(overlap)}')
report['family_overlap']=0
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
