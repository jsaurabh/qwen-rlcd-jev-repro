"""Build private typed-decision candidates; source and family provenance retained."""
from pathlib import Path
from collections import Counter,defaultdict
import json,random,hashlib,math,itertools,time
import pyarrow.parquet as pq
R=Path('/content/data-v3/work');S=R/'sources';O=R/'candidates';O.mkdir(exist_ok=True)
rng=random.Random(20261009)
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def split(f):
 n=int(digest(f)[:8],16)%100
 return 'dev' if n<5 else 'temperature' if n<10 else 'train'
files={k:(O/f'{k}.jsonl').open('w') for k in ['train','dev','temperature']};counts=Counter();families=defaultdict(set);seen=set()
def emit(source,family,state,question,target,key,fold=None):
 i=digest(source+'|'+family+'|'+key)
 if i in seen:return
 seen.add(i);fold=fold or split(family)
 row={'id':source+':'+i,'family':family,'suite':source,'state':state,'question':question,'target':target,'label':target,'source':{'dataset':source,'split':'train','family_split':fold}}
 files[fold].write(json.dumps(row,ensure_ascii=False)+'\n');counts[source+':'+fold]+=1;families[fold].add(family)
def progress(phase):
 (R/'candidate-status.json').write_text(json.dumps({'phase':phase,'counts':dict(counts),'time':time.time()}));print(json.dumps({'phase':phase,'counts':dict(counts)}),flush=True)
# Existing control splits stay separate; duplicate variants are not manufactured.
for fold in files:
 rows=[json.loads(l) for l in (R/'audit'/f'{fold}.jsonl').read_text().splitlines()]
 rng.shuffle(rows)
 for row in rows[:30000 if fold=='train' else 1000]:
  row['mix_component']='existing';files[fold].write(json.dumps(row)+'\n');counts['existing:'+fold]+=1
progress('existing_added')
# ACOS annotations are exhaustive quadruples for the released tasks.
# Use exact annotated spans; implicit span(-1,-1) is represented explicitly.
# Local research input only: source release lacks an explicit redistribution license.
for p in sorted((S/'acos').glob('*train.tsv')):
 domain=p.stem.split('_')[0]
 parsed=[];categories=set()
 for line in p.read_text().splitlines():
  parts=line.split('\t');text=parts[0];quads=[]
  for field in parts[1:]:
   a,category,sentiment,o=field.split();categories.add(category)
   def span(v):
    start,end=map(int,v.split(','));return 'implicit' if start==-1 else ' '.join(text.split()[start:end])
   quads.append({'aspect':span(a),'category':category,'opinion':span(o),'sentiment':{'0':'negative','1':'neutral','2':'positive'}[sentiment]})
  parsed.append((text,quads))
 for text,quads in parsed:
  family='review:'+digest(text.casefold());gold={json.dumps(q,sort_keys=True) for q in quads}
  for q in quads:
   variations=[q]
   for sentiment in ['negative','neutral','positive']:
    if sentiment!=q['sentiment']:variations.append({**q,'sentiment':sentiment})
   other=sorted(categories-{q['category']})
   if other:variations.append({**q,'category':rng.choice(other)})
   for candidate in variations:
    key=json.dumps(candidate,sort_keys=True);target=key in gold
    question={'type':'noul','instructions':'Does this aspect/category/opinion/sentiment combination accurately describe the review? An implicit aspect or opinion is allowed.','criteria':{'true':candidate,'false':'This combination is not supported by the review.'}}
    emit('acos-quad-verification',family,text,question,target,key)
  # A joint sentiment judgment avoids introducing unsupported negative labels.
  for q in quads:
   sentiments={x['sentiment'] for x in quads if x['aspect']==q['aspect'] and x['category']==q['category']}
   if len(sentiments)==1:
    question={'type':'choice','instructions':f"What sentiment is expressed about aspect {q['aspect']!r}, category {q['category']}?",'criteria':{k:k for k in ['negative','neutral','positive']}}
    emit('acos-sentiment',family,text,question,next(iter(sentiments)),q['aspect']+'|'+q['category'])
progress('acos_added')
# English training examples only. Family by query; discard products crossing folds.
examples=pq.read_table(S/'esci/shopping_queries_dataset_examples.parquet').to_pandas()
examples=examples[(examples['split']=='train')&(examples['product_locale']=='us')].sample(frac=1,random_state=20261009)
chosen=[];product_fold={};limits={'train':30000,'dev':1500,'temperature':1500};taken=Counter()
for row in examples.to_dict('records'):
 family='query:'+digest(str(row['query']).casefold());fold=split(family);pid=str(row['product_id'])
 if taken[fold]>=limits[fold] or (pid in product_fold and product_fold[pid]!=fold):continue
 product_fold[pid]=fold;chosen.append((row,family,fold));taken[fold]+=1
 if all(taken[k]>=v for k,v in limits.items()):break
wanted={str(r['product_id']) for r,_,_ in chosen};products={}
for batch in pq.ParquetFile(S/'esci/shopping_queries_dataset_products.parquet').iter_batches(batch_size=10000):
 for p in batch.to_pylist():
  if p['product_locale']=='us' and str(p['product_id']) in wanted:products[str(p['product_id'])]=p
for r,family,fold in chosen:
 p=products.get(str(r['product_id']))
 if not p:continue
 state={'query':r['query'],'product':{k:(p[k] if isinstance(p[k],str) else '') for k in ['product_title','product_description','product_bullet_point','product_brand','product_color']}}
 question={'type':'choice','instructions':'Classify the relevance of this product to the shopping query.','criteria':{'E':'Exact match: satisfies the query requirements.','S':'Substitute: similar but does not satisfy every requirement.','C':'Complement: useful alongside the requested item, not a substitute.','I':'Irrelevant: does not meet or complement the query.'}}
 emit('esci-relevance',family,state,question,str(r['esci_label']),str(r['example_id']),fold)
progress('esci_added')
# Verified short programs. Gold is computed by explicit formulas; no arbitrary execution.
# Template families are disjoint across train/dev/calibration.
expressions=[
 ('a + b * c',lambda a,b,c:a+b*c),
 ('(a + b) * c',lambda a,b,c:(a+b)*c),
 ('sum(range(a, a + b))',lambda a,b,c:sum(range(a,a+b))),
 ('sum(x * c for x in range(a, a + b))',lambda a,b,c:sum(x*c for x in range(a,a+b))),
 ('len([x for x in range(a, a + b) if x % c == 0])',lambda a,b,c:len([x for x in range(a,a+b) if x%c==0])),
 ('a ** 2 - b * c',lambda a,b,c:a*a-b*c),
 ('max(a, b) * c - min(a, b)',lambda a,b,c:max(a,b)*c-min(a,b)),
 ('(a * b) // c + (a * b) % c',lambda a,b,c:(a*b)//c+(a*b)%c),
 ('sum(range(b)) + a * c',lambda a,b,c:sum(range(b))+a*c),
 ('abs(a - b) + c ** 2',lambda a,b,c:abs(a-b)+c*c),
 ('sum([a, b, c]) * 2',lambda a,b,c:(a+b+c)*2),
 ('(a + c) * (b - c)',lambda a,b,c:(a+c)*(b-c)),
]
for t,(expr,fn) in enumerate(expressions):
 fold='train' if t<8 else 'dev' if t<10 else 'temperature';family=f'verified-code-template:{t}'
 limit=2000 if fold=='train' else 500
 parameters=list(itertools.product(range(1,31),range(1,31),range(1,10)));rng.shuffle(parameters)
 for a,b,c in parameters[:limit]:
  gold=fn(a,b,c);vals={gold}
  for delta in rng.sample([i for i in range(-20,21) if i],6):vals.add(gold+delta)
  vals=list(vals);rng.shuffle(vals);criteria={f'v{i}':str(v) for i,v in enumerate(vals)};target=next(k for k,v in criteria.items() if v==str(gold))
  emit('verified-code',family,f'a = {a}\nb = {b}\nc = {c}\nprint({expr})',{'type':'choice','instructions':'What integer does this Python program print?','criteria':criteria},target,f'{a},{b},{c}',fold)
progress('verified_code_added')
for f in files.values():f.close()
for x,y in [('train','dev'),('train','temperature'),('dev','temperature')]:assert not families[x]&families[y]
report={'counts':dict(counts),'new_family_overlap':0,'proposed_target':100000,'status':'candidates_only_not_training_ready','changes_from_proposal':'xLAM access unavailable; Tulu and Numina withheld pending trustworthy decision-label conversion. Reallocated toward ESCI and programmatically verified code. No artificial duplication to force100k.','limitations':['ACOS original research release has no explicit license file; no raw redistribution.','Original candidates retain their historical split; new query/product and review families separate.','New candidates still require benchmark overlap screen, token-length audit and held-out cohort freeze.'],'files':{k:{'sha256':hashlib.sha256((O/f'{k}.jsonl').read_bytes()).hexdigest()} for k in files}}
(O/'candidate-manifest.json').write_text(json.dumps(report,indent=2));(O/'COMPLETE.json').write_text(json.dumps({'time':time.time()}));progress('candidates_complete')
