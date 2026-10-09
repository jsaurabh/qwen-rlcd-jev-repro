"""Private, conservative state-overlap screen. Outputs aggregate evidence only."""
from pathlib import Path
from collections import Counter
import json,gzip,re,unicodedata,hashlib,time
from datasketch import MinHash,MinHashLSH
R=Path('/content/data-v3');W=R/'work';O=W/'audit';O.mkdir(exist_ok=True)
def status(phase,**kw):
 (W/'status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)));print(json.dumps(dict(phase=phase,**kw)),flush=True)
def normalize(v):
 if not isinstance(v,str):v=json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
 return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',v).casefold()))
def state(r):
 if 'state' in r:return r['state']
 for k in ['input','request','payload']:
  if isinstance(r.get(k),dict) and 'state' in r[k]:return r[k]['state']
 raise ValueError('Missing state in row; refusing incomplete audit')
def shingle(s):
 t=s.split()[:4096]
 return set(' '.join(t[i:i+5]).encode() for i in range(max(0,len(t)-4)))
def sig(tokens):
 m=MinHash(num_perm=64,seed=20261009);m.update_batch(list(tokens));return m
exact=set();lsh=MinHashLSH(threshold=.75,num_perm=64);texts={};seen=set();counter=Counter()
status('indexing_benchmark')
for p in sorted((W/'suite-0.3').glob('*rows.jsonl.gz')):
 with gzip.open(p,'rt') as f:
  for line in f:
   row=json.loads(line);s=normalize(state(row));key=hashlib.sha256(s.encode()).hexdigest()
   exact.add(key)
   if key in seen:continue
   seen.add(key);tokens=shingle(s)
   if len(tokens)>=10:
    lsh.insert(key,sig(tokens));texts[key]=tokens
   counter['benchmark_unique_states']+=1
   if len(seen)%5000==0:status('indexing_benchmark',unique_states=len(seen))
report={'method':'Normalized state SHA256 plus MinHash64 LSH .75 candidate retrieval with exact 5-word shingle Jaccard>=.8. Near-match text capped at4096words. Exact whole-state hash is uncapped. This screen is not proof of absence of semantic contamination.','benchmark_unique_states':len(seen),'splits':{}}
for split in ['train','dev','temperature']:
 counts=Counter();suites=Counter();kept=0
 with (W/'original'/f'{split}.jsonl').open() as f,(O/f'{split}.jsonl').open('w') as out:
  for i,line in enumerate(f):
   row=json.loads(line);s=normalize(state(row));key=hashlib.sha256(s.encode()).hexdigest();reason=None
   if key in exact:reason='exact_state'
   else:
    t=shingle(s)
    if len(t)>=10:
     for k in lsh.query(sig(t)):
      u=texts[k]
      if len(t&u)/len(t|u)>=.8:reason='near_state';break
   counts['input']+=1
   if reason:counts[reason]+=1;suites[str(row.get('suite','unknown'))]+=1
   else:out.write(line);kept+=1
   if i%2000==0:status('auditing',split=split,processed=i,kept=kept)
 report['splits'][split]={'counts':dict(counts),'kept':kept,'removed_by_suite':dict(suites),'sha256':hashlib.sha256((O/f'{split}.jsonl').read_bytes()).hexdigest()}
(O/'audit-report.json').write_text(json.dumps(report,indent=2))
(O/'COMPLETE.json').write_text(json.dumps({'phase':'overlap_screen_complete','time':time.time()}))
status('audit_complete',report=report)
