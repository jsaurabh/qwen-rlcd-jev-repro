"""Small, independent MLX decision-training reconstruction; not TypeSafe's RLCD."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import httpx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers

SCHEMA = [
    {
        "name": "department",
        "type": "choice",
        "question": (
            "Which team should handle this case? "
            "Payments handles transfers, charges, and payment status. "
            "Account support handles access and account servicing. "
            "Fraud review handles suspected unauthorized activity."
        ),
        "choices": ["payments", "account_support", "fraud_review"],
    },
    {
        "name": "urgency",
        "type": "score",
        "question": (
            "Rate urgency: "
            "0 = routine inquiry with no immediate disruption; "
            "1 = access or payment disruption needing prompt attention; "
            "2 = suspected ongoing unauthorized activity or imminent loss."
        ),
        "choices": [0, 1, 2],
    },
    {
        "name": "escalate",
        "type": "noul",
        "question": (
            "Does this case need human review because it involves "
            "suspected fraud, disputed authorization, conflicting "
            "information, or an exception to the stated bank policy?"
        ),
        "choices": [False, True],
    },
]

MODEL = 'models/qwen-0.5b-4bit'
LORA = {'rank': 4, 'scale': 8.0, 'dropout': 0.0, 'keys': ['self_attn.q_proj', 'self_attn.v_proj']}
SCORE_RUBRIC = [
    'Routine inquiry with no immediate disruption',
    'Access or payment disruption needing prompt attention',
    'Suspected ongoing unauthorized activity or imminent loss',
]


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def read_typesafe_key():
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    env_path = Path(__file__).resolve().parent / '.env'
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('TYPESAFE_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit('TYPESAFE_API_KEY is missing from the environment and project .env')
    return key


def typesafe_questions():
    return {
        'department': {
            'type': 'choice',
            'instructions': 'Which team should handle this banking case?',
            'criteria': {
                'payments': 'Transfers, charges, payment processing, and payment status',
                'account_support': 'Account access, identity, profile, statements, and account servicing',
                'fraud_review': 'Suspected unauthorized access, payments, transfers, or withdrawals',
            },
        },
        'urgency': {
            'type': 'score',
            'instructions': SCHEMA[1]['question'],
            'criteria': SCORE_RUBRIC,
        },
        'escalate': {
            'type': 'noul',
            'instructions': SCHEMA[2]['question'],
            'criteria': {
                'true': 'Human review is required under the stated conditions',
                'false': 'No human review is required under the stated conditions',
            },
        },
    }


def comparison_scores(local_scores):
    """Give local and Jev results a common, semantically honest shape."""
    return {
        name: {
            'type': field['type'],
            'value': field['value'],
            'top_choice': field['top_choice'],
            'top_probability': field['confidence'],
            'provider_confidence': None,
            'distribution': field['distribution'],
        }
        for name, field in local_scores.items()
    }


def jev_decide(state, model_name):
    started = time.perf_counter()
    with httpx.Client(
        base_url='https://api.typesafe.ai',
        headers={'Authorization': 'Bearer ' + read_typesafe_key()},
        timeout=30.0,
        follow_redirects=False,
    ) as client:
        response = client.post('/v1/systemone', json={
            'state': state,
            'model': model_name,
            'questions': typesafe_questions(),
        })
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        # Avoid printing response bodies, which may include sensitive request data.
        raise RuntimeError(f'Jev HTTP {response.status_code}')
    raw = response.json()
    answers = raw['answers']
    distributions = [
        [answers['department']['probabilities'][choice] for choice in SCHEMA[0]['choices']],
        [answers['urgency']['probabilities'][str(level)] for level in SCHEMA[1]['choices']],
        [1.0 - answers['escalate']['noul'], answers['escalate']['noul']],
    ]
    result = {}
    for spec, probabilities in zip(SCHEMA, distributions):
        if len(probabilities) != len(spec['choices']) or min(probabilities) < 0 or max(probabilities) > 1:
            raise ValueError('Jev returned an invalid candidate distribution')
        total = sum(probabilities)
        if abs(total - 1.0) > 0.002:
            raise ValueError('Jev probabilities do not sum to one')
        probabilities = np.asarray(probabilities, dtype=float) / total
        best = int(probabilities.argmax())
        value = spec['choices'][best]
        if spec['type'] == 'score':
            value = float(np.dot(probabilities, spec['choices']))
        elif spec['type'] == 'noul':
            value = float(probabilities[1])
        result[spec['name']] = {
            'type': spec['type'],
            'value': value,
            'top_choice': spec['choices'][best],
            'top_probability': float(probabilities[best]),
            # TypeSafe defines this separately from the maximum probability.
            # Noul intentionally has no provider confidence field.
            'provider_confidence': answers[spec['name']].get('confidence'),
            'distribution': [
                {'choice': choice, 'probability': float(probability)}
                for choice, probability in zip(spec['choices'], probabilities)
            ],
        }
    return {
        'model': raw.get('model', model_name),
        'elapsed_ms': elapsed_ms,
        'usage': raw.get('usage'),
        'scores': result,
    }


def scored_summary(records, key):
    """Summarize labeled typed decisions from either comparison backend."""
    correct, confidences, nll, brier = [], [], [], []
    per_field = {spec['name']: [] for spec in SCHEMA}
    for record in records:
        scores = record[key]['scores']
        for index, spec in enumerate(SCHEMA):
            field = scores[spec['name']]
            probabilities = np.asarray(
                [item['probability'] for item in field['distribution']], dtype=float
            )
            probabilities /= probabilities.sum()
            label = record['labels'][index]
            prediction = int(probabilities.argmax())
            hit = float(prediction == label)
            correct.append(hit)
            per_field[spec['name']].append(hit)
            confidences.append(float(probabilities.max()))
            nll.append(-math.log(max(float(probabilities[label]), 1e-12)))
            target = np.eye(len(probabilities))[label]
            brier.append(float(np.sum((probabilities - target) ** 2)))
    correct = np.asarray(correct)
    confidences = np.asarray(confidences)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    return {
        'cases': len(records),
        'decisions': len(correct),
        'accuracy': float(correct.mean()),
        'nll': float(np.mean(nll)),
        'brier': float(np.mean(brier)),
        'ece_5_bins': ece,
        'field_accuracy': {
            name: float(np.mean(values)) for name, values in per_field.items()
        },
    }


def compare_external_local_with_jev(local_path, model_name, output_path):
    """Compare externally scored local records with Jev on identical inputs."""
    local = json.loads(Path(local_path).read_text())
    records = []
    for index, source in enumerate(local['records'], 1):
        try:
            jev = jev_decide(source['state'], model_name)
        except Exception:
            # Avoid retrying potentially charged calls or serializing provider bodies.
            raise RuntimeError(f'Jev comparison failed at case {index}') from None
        records.append({
            'id': source['id'],
            'state': source['state'],
            'labels': source['labels'],
            'local_qwen': {'scores': source['scores']},
            'jev': jev,
        })
        print(f'Jev {index}/{len(local["records"])} ok', flush=True)
    payload = {
        'schema': SCHEMA,
        'local_backend': {
            key: local.get(key) for key in
            ('backend', 'adapter', 'base', 'base_revision', 'elapsed_s')
        },
        'jev_model': model_name,
        'summaries': {
            'local_qwen': scored_summary(records, 'local_qwen'),
            'jev': scored_summary(records, 'jev'),
        },
        'records': records,
    }
    dump(output_path, payload)
    print(json.dumps(payload['summaries'], indent=2), flush=True)
    return payload


def make_data(n, seed, offset):
    rng = random.Random(seed)
    # Illustrative routing labels, not institutional policy or real customer data.
    cases = [
        ('customer asks when a scheduled transfer will arrive', 0, 0, False),
        ('customer requests a statement explaining a recognized charge', 0, 0, False),
        ('a recognized payment is delayed and blocking a purchase', 0, 1, False),
        ('customer requests an exception to a transfer policy', 0, 1, True),
        ('customer asks how to update their mailing address', 1, 0, False),
        ('customer cannot sign in after forgetting their password', 1, 1, False),
        ('account ownership information conflicts across submitted records', 1, 1, True),
        ('customer disputes authorization of a completed card charge', 2, 1, True),
        ('unrecognized transfers are occurring and another transfer is pending', 2, 2, True),
        ('stolen credentials are being used to move funds now', 2, 2, True),
    ]
    rows = []
    for i in range(n):
        issue, dep, urgency, review = rng.choice(cases)
        ambiguous = rng.random() < .25
        state = f'Bank case {offset+i}: {issue}. ' + ('Report is unverified.' if ambiguous else 'Report is confirmed.')
        # Known synthetic posterior, not a claim about real support-ticket uncertainty.
        certainty = .60 if ambiguous else .90
        d = [(1-certainty)/2] * 3
        d[dep] = certainty
        u = [(1-certainty)/2] * 3
        u[urgency] = certainty
        e = .85 if review else .15
        if ambiguous:
            e = .5
        targets = [d, u, [1-e, e]]
        labels = [rng.choices(range(len(q)), weights=q)[0] for q in targets]
        rows.append({'state': state, 'targets': targets, 'labels': labels})
    return rows


class DecisionEngine:
    def __init__(self, model, tokenizer, max_length=256):
        self.model, self.tokenizer, self.max_length = model, tokenizer, max_length
        self.candidates = []
        for s in SCHEMA:
            ids = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(len(s['choices']))]
            if any(len(t) != 1 for t in ids) or len({t[0] for t in ids}) != len(ids):
                raise ValueError('Candidate aliases must be distinct single tokens')
            self.candidates.append(mx.array([t[0] for t in ids]))

    def encode(self, state):
        catalog = '\n'.join(s['name'] + ': ' + s['question'] + ' ' + ', '.join(f'{chr(65+i)}={json.dumps(v)}' for i,v in enumerate(s['choices'])) for s in SCHEMA)
        text = self.tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'Evaluate each question independently. Return only its option letter.'},
            {'role': 'user', 'content': 'State: '+state+'\nSchema:\n'+catalog},
        ], tokenize=False, add_generation_prompt=True)
        # Tokenize prefix and suffix separately in BOTH training and cached inference.
        prefix = self.tokenizer.encode(text, add_special_tokens=False)
        suffixes = [self.tokenizer.encode(s['name'] + ' option:', add_special_tokens=False) for s in SCHEMA]
        if max(len(prefix)+len(s) for s in suffixes) > self.max_length:
            raise ValueError('Prompt exceeds max length; shorten state/schema (never silently truncate)')
        return prefix, suffixes

    def project(self, hidden):
        # Avoid the enormous [batch, sequence, vocabulary] training tensor.
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)

    def logits(self, ids, candidates):
        hidden = self.model.model(ids)[:, -1, :]
        return self.project(hidden)[0, candidates].astype(mx.float32)

    def parallel_logits(self, state, suffix_override=None):
        prefix, suffixes = self.encode(state)
        if suffix_override is not None:
            suffixes = suffix_override
        cache = make_prompt_cache(self.model)
        h = self.model.model(mx.array([prefix]), cache=cache)
        mx.eval(h, [c.state for c in cache])
        branches = []
        for c in cache:
            new = copy.copy(c)
            new.keys = mx.repeat(c.keys, len(SCHEMA), axis=0)
            new.values = mx.repeat(c.values, len(SCHEMA), axis=0)
            branches.append(new)
        longest = max(map(len, suffixes))
        padded = [s + [self.tokenizer.pad_token_id]*(longest-len(s)) for s in suffixes]
        h = self.model.model(mx.array(padded), cache=branches)
        # Right padding is never read and cannot affect earlier positions causally.
        h = h[mx.arange(len(SCHEMA)), mx.array([len(s)-1 for s in suffixes])]
        all_logits = self.project(h)
        result = [all_logits[i, candidates].astype(mx.float32) for i,candidates in enumerate(self.candidates)]
        mx.eval(result)
        return result

    def decide(self, state, temperature=1.):
        result = {}
        for s,z in zip(SCHEMA,self.parallel_logits(state)):
            p = np.array(mx.softmax(z/temperature).tolist())
            best = int(p.argmax())
            value = s['choices'][best]
            if s['type'] == 'score':
                value = float(np.dot(p, s['choices']))
            if s['type'] == 'noul':
                value = float(p[1])
            result[s['name']] = {'type': s['type'], 'value': value, 'top_choice': s['choices'][best], 'confidence': float(p[best]), 'distribution': [{'choice': c, 'probability': float(v)} for c,v in zip(s['choices'],p)]}
        return result


def collect(engine, rows):
    return [[np.array(z.tolist(), dtype=np.float64) for z in engine.parallel_logits(r['state'])] for r in rows]


def metrics(logits, rows, temperature=1.):
    nll, brier, ce, mse, correct, conf = [], [], [], [], [], []
    for zs,row in zip(logits,rows):
        for z,q,y in zip(zs,row['targets'],row['labels']):
            z = z/temperature
            p = np.exp(z-z.max()); p /= p.sum()
            target = np.eye(len(p))[y]
            nll.append(-math.log(max(p[y],1e-12)))
            brier.append(float(np.sum((p-target)**2)))
            ce.append(float(-np.dot(q,np.log(np.maximum(p,1e-12)))))
            mse.append(float(np.sum((p-np.array(q))**2)))
            correct.append(float(p.argmax()==y)); conf.append(float(p.max()))
    correct, conf = np.array(correct), np.array(conf)
    ece = 0.
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any(): ece += float(mask.mean()*abs(correct[mask].mean()-conf[mask].mean()))
    return {'decisions':len(nll), 'accuracy_sampled':float(correct.mean()), 'nll_sampled':float(np.mean(nll)), 'brier_sampled':float(np.mean(brier)), 'ece_5_bins_sampled':ece, 'cross_entropy_known_posterior':float(np.mean(ce)), 'squared_error_known_posterior':float(np.mean(mse))}


def checks(engine, state):
    prefix,suffixes = engine.encode(state)
    shared = engine.parallel_logits(state)
    diffs = []
    for i,suffix in enumerate(suffixes):
        full = engine.logits(mx.array([prefix+suffix]),engine.candidates[i])
        diffs.append(float(mx.max(mx.abs(mx.softmax(full)-mx.softmax(shared[i])))))
    assert max(diffs)<.015, diffs  # fp16 batched Metal kernels need not be bit-identical.
    # Exact expected bandit gradient equals half Brier gradient in logits space.
    p,q = np.array([.2,.3,.5]),np.array([.1,.6,.3])
    jac = np.diag(p)-np.outer(p,p)
    expected = sum(p[a]*(q[a]-p[a])*(np.eye(3)[a]-p) for a in range(3))
    assert np.allclose(expected, jac@(q-p))
    changed=list(suffixes)
    changed[-1]=engine.tokenizer.encode('escalate different question option:',add_special_tokens=False)
    independent=engine.parallel_logits(state,changed)
    independence_error=max(float(mx.max(mx.abs(mx.softmax(a)-mx.softmax(b)))) for a,b in zip(shared[:-1],independent[:-1]))
    assert independence_error<.015
    output=engine.decide(state)
    assert isinstance(output['department']['value'],str)
    assert 0<=output['urgency']['value']<=2 and 0<=output['escalate']['value']<=1
    for field in output.values():
        assert abs(sum(v['probability'] for v in field['distribution'])-1)<1e-5
    return {'cached_vs_full_max_probability_error':max(diffs),'other_suffix_independence_probability_error':independence_error,'proper_reward_gradient_identity':True,'typed_outputs_and_normalization':True}


def train(engine, rows, args, out):
    model=engine.model
    model.freeze()
    linear_to_lora_layers(model,args.lora_layers,LORA)
    initial={k:np.array(v.tolist()) for k,v in tree_flatten(model.trainable_parameters())}
    assert initial and all('lora_' in k for k in initial), list(initial)
    model.train()
    optimizer=optim.Adam(learning_rate=args.lr)
    rng=np.random.default_rng(args.seed)
    examples=[]
    for r in rows:
        pre,sufs=engine.encode(r['state'])
        for i,s in enumerate(sufs):
            examples.append((mx.array([pre+s]),engine.candidates[i],mx.array(r['targets'][i])))
    def loss_fn(m, x, candidates, target, action, reward, bandit):
        z=engine.logits(x,candidates)
        logp=z-mx.logsumexp(z)
        if bandit:
            return -mx.stop_gradient(reward)*logp[action]
        p=mx.exp(logp)
        return -mx.sum(target*logp)+.5*mx.sum((p-target)**2)
    vg=nn.value_and_grad(model,loss_fn)
    history=[]
    start=time.perf_counter()
    for step in range(args.steps+args.rl_steps):
        bandit=step>=args.steps
        acc=None; total=0.
        for _ in range(args.accum):
            x,c,q=examples[int(rng.integers(len(examples)))]
            action,reward=0,mx.array(0.)
            if bandit:
                p=np.array(mx.softmax(engine.logits(x,c)).tolist(),dtype=np.float64);p/=p.sum()
                action=int(rng.choice(len(p),p=p))
                # Environment sees q; trainer receives only correctness of its chosen action.
                q_env=np.array(q.tolist(),dtype=np.float64); q_env/=q_env.sum()
                truth=int(rng.choice(len(p),p=q_env))
                reward=mx.array(float(action==truth)-p[action],dtype=mx.float32)
            loss,grad=vg(model,x,c,q,action,reward,bandit)
            mx.eval(loss,grad)
            assert math.isfinite(float(loss)), 'Non-finite loss'
            acc=grad if acc is None else tree_map(lambda a,b:a+b,acc,grad)
            mx.eval(acc)
            total+=float(loss)
        acc=tree_map(lambda g:g/args.accum,acc)
        acc,norm=optim.clip_grad_norm(acc,1.)
        assert math.isfinite(float(norm)), 'Non-finite gradient'
        optimizer.update(model,acc)
        mx.eval(model.parameters(),optimizer.state)
        entry={'step':step+1,'phase':'bandit' if bandit else 'supervised','loss':total/args.accum,'grad_norm':float(norm),'elapsed_seconds':time.perf_counter()-start}
        history.append(entry);print(json.dumps(entry),flush=True)
        if step+1==args.steps:
            mx.save_safetensors(str(out/'warmup.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    model.eval()
    mx.save_safetensors(str(out/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
    dump(out/'adapter_config.json',{'fine_tune_type':'lora','num_layers':args.lora_layers,'lora_parameters':LORA})
    dump(out/'history.json',history)
    delta=sum(float(np.sum((np.array(v.tolist())-initial[k])**2)) for k,v in tree_flatten(model.trainable_parameters()))**.5
    assert delta>0, 'Adapter weights did not change'
    return {'adapter_update_l2':delta,'seconds':time.perf_counter()-start,'trainable_parameters':sum(x.size for _,x in tree_flatten(model.trainable_parameters())),'updates':len(history)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default=MODEL)
    ap.add_argument('--out',default='runs/smoke')
    ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--rl-steps',type=int,default=4)
    ap.add_argument('--accum',type=int,default=4)
    ap.add_argument('--lora-layers',type=int,default=4)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--train-size',type=int,default=32)
    ap.add_argument('--eval-size',type=int,default=8)
    ap.add_argument('--max-length',type=int,default=256)
    ap.add_argument('--adapter',default=None)
    ap.add_argument('--state',default=None)
    ap.add_argument('--compare-jev',action='store_true',help='Call Jev with the same state and return scores from both backends')
    ap.add_argument('--jev-model',default='jev-latest')
    ap.add_argument('--local-results',default=None,help='JSON predictions from the PyTorch/Colab adapter bridge')
    ap.add_argument('--compare-out',default='runs/hf-adapter-jev-live/comparison.json')
    args=ap.parse_args()
    wall_start=time.perf_counter()
    if min(args.accum,args.train_size,args.eval_size,args.lora_layers)<1 or args.steps<1 or args.rl_steps<0: ap.error('Invalid sizes')
    mx.random.seed(args.seed)
    mx.set_cache_limit(512*1024**2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.local_results:
        compare_external_local_with_jev(
            args.local_results, args.jev_model, args.compare_out
        )
        return
    model,tok=load(args.model,adapter_path=args.adapter)
    engine=DecisionEngine(model,tok,args.max_length)
    model.eval()
    if args.state:
        temperature=1.
        if args.adapter and (Path(args.adapter)/'calibration.json').exists():
            temperature=json.loads((Path(args.adapter)/'calibration.json').read_text())['temperature']
        started=time.perf_counter()
        local_scores=engine.decide(args.state,temperature)
        local_elapsed_ms=(time.perf_counter()-started)*1000
        if not args.compare_jev:
            print(json.dumps(local_scores,indent=2));return
        jev=jev_decide(args.state,args.jev_model)
        output={
            'state':args.state,
            'local_qwen':{
                'model':args.model,
                'adapter':args.adapter,
                'temperature':temperature,
                'elapsed_ms':local_elapsed_ms,
                'scores':comparison_scores(local_scores),
            },
            'jev':jev,
        }
        print(json.dumps(output,indent=2));return
    train_rows=make_data(args.train_size,args.seed,0)
    val=make_data(args.eval_size,args.seed+1,10000)
    test=make_data(args.eval_size,args.seed+2,20000)
    for name,rows in [('train',train_rows),('validation',val),('test',test)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report={'config':vars(args),'schema':SCHEMA,'checks_before':checks(engine,test[0]['state'])}
    before=collect(engine,test)
    report['before']=metrics(before,test)
    report['training']=train(engine,train_rows,args,out)
    report['checks_after']=checks(engine,test[0]['state'])
    final_test=collect(engine,test)
    val_logits=collect(engine,val)
    grid=np.geomspace(.25,4.,81)
    t=float(min(grid,key=lambda t:metrics(val_logits,val,t)['cross_entropy_known_posterior']))
    dump(out/'calibration.json',{'temperature':t,'fit_split':'validation','criterion':'cross entropy to known synthetic posterior','grid':[.25,4.,81]})
    report['after']=metrics(final_test,test)
    report['after_temperature']=metrics(final_test,test,t)
    report['temperature']=t
    # Evaluate warmup separately, restoring the final adapter afterwards.
    model.load_weights(str(out/'warmup.safetensors'),strict=False)
    report['after_supervised']=metrics(collect(engine,test),test)
    model.load_weights(str(out/'adapters.safetensors'),strict=False)
    reloaded=collect(engine,test[:1])[0]
    reload_error=max(float(np.max(np.abs(a-b))) for a,b in zip(reloaded,final_test[0]))
    assert reload_error<1e-5
    report['checks_after']['adapter_reload_max_logit_error']=reload_error
    latencies=[]
    for _ in range(5):
        t0=time.perf_counter();engine.parallel_logits(test[0]['state']);latencies.append((time.perf_counter()-t0)*1000)
    report['warm_three_field_inference_ms_median']=float(np.median(latencies))
    report['wall_seconds']=time.perf_counter()-wall_start
    report['peak_mlx_gb']=mx.get_peak_memory()/1e9
    report['max_prompt_tokens']=max(len(engine.encode(r['state'])[0])+max(map(len,engine.encode(r['state'])[1])) for r in train_rows+val+test)
    report['data_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}
    report['caveat']='Smoke test only; tiny related synthetic splits cannot demonstrate general calibration or Jev equivalence.'
    dump(out/'report.json',report)
    dump(out/'predictions.json',{'state':test[0]['state'],'result':engine.decide(test[0]['state'],t)})
    dump(out/'test_logits.json',{'before':[[z.tolist() for z in zs] for zs in before],'after':[[z.tolist() for z in zs] for zs in final_test]})
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
