"""Conservative private benchmark and cross-fold document overlap screen."""
from pathlib import Path
from collections import Counter
import json,gzip,re,hashlib,unicodedata,time
from datasketch import MinHash,MinHashLSH
R=Path('/content/data-v3/work');I=R/'candidates';O=R/'screened';O.mkdir(exist_ok=True)
def norm(v):
 if not isinstance(v,str):v=json.dumps(v,sort_keys=True,ensure_ascii=False)
 return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',v).casefold()))
def parts(v):
 result={norm(v)}
 def walk(x):
  if isinstance(x,str):
   s=norm(x)
   if len(s.split())>=12:result.add(s)
  elif isinstance(x,dict):
   for value in x.values():walk(value)
  elif isinstance(x,list):
   for value in x:walk(value)
 walk(v);return {x for x in result if x}
def state(r):
 if 'state' in r:return r['state']
 for k in ['input','request','payload']:
  if isinstance(r.get(k),dict) and 'state' in r[k]:return r[k]['state']
 raise ValueError('Missing state')
def h(t):return hashlib.sha256(t.encode()).hexdigest()
def shingles(s):
 t=s.split()[:4096];return set(' '.join(t[i:i+5]).encode() for i in range(max(0,len(t)-4)))
def signature(t):
 m=MinHash(num_perm=64,seed=20261009);m.update_batch(list(t));return m
class Index:
 def __init__(self):self.exact=set();self.sets={};self.lsh=MinHashLSH(threshold=.75,num_perm=64)
 def add(self,v):
  for s in parts(v):
   k=h(s)
   if k in self.exact:continue
   self.exact.add(k);t=shingles(s)
   if len(t)>=10:self.sets[k]=t;self.lsh.insert(k,signature(t))
 def match(self,v):
  for s in parts(v):
   if h(s) in self.exact:return 'exact_document'
   t=shingles(s)
   if len(t)<10:continue
   for k in self.lsh.query(signature(t)):
    u=self.sets[k]
    if len(t&u)/len(t|u)>=.8:return 'near_document'
  return None
def status(phase,**kw):
 (R/'screen-status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw)));print(json.dumps(dict(phase=phase,**kw)),flush=True)
# Planted positive/negative controls test nested wrapping and normalization.
test=Index();text='A customer reports that an unexpected duplicate monthly subscription payment appeared on their account yesterday.'
test.add({'document':text,'other':'metadata'});assert test.match(text)=='exact_document'
assert test.match(text.upper())=='exact_document'
assert test.match('Unrelated astronomy observations describe stars and galaxies.') is None
bench=Index();status('benchmark_index')
for p in sorted((R/'suite-0.3').glob('*rows.jsonl.gz')):
 with gzip.open(p,'rt') as f:
  for i,l in enumerate(f):
   bench.add(state(json.loads(l)))
   if i%10000==0:status('benchmark_index',documents=len(bench.exact))
report={'method':'Whole normalized states and nested string fields of12+words; exact hashes and MinHash64 with exact5gram Jaccard>=.8 (first4096words for near matching). Conservative document-level removal, including cross-fold collisions. Not a proof against semantic or pretraining contamination.','positive_negative_controls':'passed','benchmark_documents':len(bench.exact),'counts':{}}
heldout=Index()
# Freeze shared calibration first, then shared development; neither sees benchmark matches.
for fold in ['temperature','dev']:
 count=Counter();rows=[];ids=set()
 for parent in [I,R/'original']:
  for line in (parent/f'{fold}.jsonl').read_text().splitlines():
   row=json.loads(line)
   if row['id'] in ids:continue
   ids.add(row['id']);count['input']+=1
   reason=bench.match(state(row))
   if reason:count['benchmark_'+reason]+=1;continue
   # Reject overlap with calibration when constructing development.
   if fold=='dev' and heldout.match(state(row)):count['calibration_overlap']+=1;continue
   rows.append(row)
 for row in rows:heldout.add(state(row))
 (O/f'{fold}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));count['kept']=len(rows);report['counts'][fold]=dict(count)
 status('heldout_screened',fold=fold,counts=dict(count))
for name,parent in [('targeted',I),('control',R/'original')]:
 count=Counter();seen=set();suites=Counter()
 with (parent/'train.jsonl').open() as f,(O/f'{name}-train.jsonl').open('w') as out:
  for i,line in enumerate(f):
   row=json.loads(line);count['input']+=1;reason=bench.match(state(row))
   if reason:count['benchmark_'+reason]+=1;continue
   if heldout.match(state(row)):count['heldout_document_overlap']+=1;continue
   key=h(norm({'state':state(row),'question':row['question']}))
   if key in seen:count['duplicate_decision']+=1;continue
   seen.add(key);out.write(line);count['kept']+=1;suites[row.get('suite','unknown')]+=1
   if i%2000==0:status('screening_train',arm=name,counts=dict(count))
 report['counts'][name]={'summary':dict(count),'kept_by_suite':dict(suites)}
(O/'screen-report.json').write_text(json.dumps(report,indent=2));(O/'COMPLETE.json').write_text(json.dumps({'time':time.time()}));status('screen_complete',report=report)
