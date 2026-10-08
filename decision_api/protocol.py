"""Explicit text-only interoperability subset, without silently dropping fields."""
import json
import math


def _fields(obj, allowed, required):
    if not isinstance(obj, dict) or set(obj) - set(allowed) or set(required) - set(obj):
        raise ValueError(f'Expected fields {sorted(required)}; allowed fields {sorted(allowed)}')


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected a nonempty string')
    return value


def _content(value):
    if not isinstance(value, (str, dict, list)):
        raise ValueError('Expected text or JSON object/array')
    json.dumps(value, allow_nan=False)
    def check(v):
        if isinstance(v, dict):
            if v.get('type') in ('image', 'image_url', 'input_image'):
                raise ValueError('This server supports text/JSON only; image inputs are not implemented')
            for x in v.values(): check(x)
        elif isinstance(v, list):
            for x in v: check(x)
    check(value)


def validate_systemone(body):
    _fields(body, ('model','state','questions'), ('model','state','questions'))
    _text(body['model']); _content(body['state'])
    questions = body['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError('Provide 1–64 named questions')
    for name, q in questions.items():
        _text(name)
        _fields(q, ('type','instructions','criteria'), ('type','instructions'))
        _content(q['instructions'])
        kind, criteria = q['type'], q.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError('Choice requires 2–255 string-keyed criteria')
            for label, description in criteria.items():
                _text(label)
                if description is not None: _content(description)
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError('Score requires 2–10 ordered criteria')
            for description in criteria: _content(description)
        elif kind == 'noul':
            if criteria is not None:
                _fields(criteria, ('false','true'), ())
                for description in criteria.values(): _content(description)
        else:
            raise ValueError('Question type must be choice, score, or noul')
    return body


def decisions_to_systemone(body):
    _fields(body, ('model','input','questions'), ('model','input','questions'))
    _text(body['model'])
    if not isinstance(body['input'], str):
        raise ValueError('The interoperability subset accepts input as a text string only')
    if not isinstance(body['questions'], list) or not 1 <= len(body['questions']) <= 64:
        raise ValueError('Provide 1–64 questions')
    mapped, mapping, names = {}, [], set()
    for index, q in enumerate(body['questions']):
        _fields(q, ('type','name','instructions','choices','levels'), ('type','instructions'))
        _text(q['instructions'])
        name = q.get('name')
        if 'name' in q:
            _text(name)
            if name in names: raise ValueError('Question names must be unique')
            names.add(name)
        key = f'q{index}'
        kind = q['type']
        internal = {'type': 'noul' if kind == 'predicate' else kind, 'instructions': q['instructions']}
        meta = {'key':key,'name':name,'type':kind}
        if kind == 'predicate':
            if 'choices' in q or 'levels' in q: raise ValueError('Predicate has no choices/levels')
        elif kind == 'choice':
            if 'levels' in q: raise ValueError('Choice has no levels')
            opts = q.get('choices')
            if not isinstance(opts, list) or not 2 <= len(opts) <= 255:
                raise ValueError('Provide 2–255 choices')
            values, descriptions, seen = [], [], set()
            for option in opts:
                _fields(option, ('value','description'), ('value',))
                value = option['value']
                if type(value) not in (str, bool): raise ValueError('Choice values must be strings or booleans')
                identity = (type(value), value)
                if identity in seen: raise ValueError('Choice values must be unique, including their type')
                seen.add(identity); values.append(value)
                description = option.get('description')
                if description is not None and not isinstance(description,str): raise ValueError('Description must be a string')
                descriptions.append(description)
            # String choices keep their original keys/prompts. Typed booleans use
            # unique aliases so True and the literal string "true" cannot collide.
            keys = values if all(type(v) is str and v for v in values) else [f'option_{i}' for i in range(len(values))]
            internal['criteria'] = {k:d if keys == values else f'{json.dumps(v)}' + (f': {d}' if d else '')
                                    for k,v,d in zip(keys,values,descriptions)}
            meta.update(values=values, keys=keys)
        elif kind == 'score':
            if 'choices' in q: raise ValueError('Score has no choices')
            levels = q.get('levels')
            if not isinstance(levels,list) or not 2 <= len(levels) <= 10: raise ValueError('Provide 2–10 levels')
            criteria, labels = [], []
            for level in levels:
                _fields(level, ('label','description'), ('label',))
                label = _text(level['label']); description = level.get('description')
                if description is not None and not isinstance(description,str): raise ValueError('Description must be a string')
                labels.append(label); criteria.append(label + (f': {description}' if description else ''))
            internal['criteria'] = criteria; meta['labels'] = labels
        else:
            raise ValueError('Question type must be predicate, choice, or score')
        mapped[key] = internal; mapping.append(meta)
    payload = {'model':body['model'],'state':body['input'],'questions':mapped}
    validate_systemone(payload)
    return payload, mapping


def systemone_to_decisions(response, mapping):
    answers=[]
    for meta in mapping:
        source=response['answers'][meta['key']]
        result={'name':meta['name'],'type':meta['type']}
        if source['type']=='refusal': result['type']='refusal'
        elif meta['type']=='predicate': result['probability']=source['noul']
        elif meta['type']=='choice':
            result.update(choice=meta['values'][meta['keys'].index(source['choice'])], confidence=source['confidence'],
                          probabilities=[{'value':v,'probability':source['probabilities'][k]} for k,v in zip(meta['keys'],meta['values'])])
        else:
            result.update(score=source['score'],confidence=source['confidence'],probabilities=[
                {'value':i,'label':label,'probability':source['probabilities'][str(i)]} for i,label in enumerate(meta['labels'])])
        answers.append(result)
    return {'model':response['model'],'answers':answers,'usage':response.get('usage')}


def answer(question, probabilities):
    kind=question['type']
    labels=['false','true'] if kind=='noul' else list(question['criteria']) if kind=='choice' else [str(i) for i in range(len(question['criteria']))]
    p=[float(v) for v in probabilities]
    if len(p)!=len(labels) or any(not math.isfinite(v) or v<0 for v in p) or not math.isclose(sum(p),1,abs_tol=1e-4):
        raise RuntimeError('Backend returned an invalid probability distribution')
    p=[v/sum(p) for v in p]
    if kind=='noul':return {'type':'noul','noul':p[1]}
    best=max(range(len(p)),key=p.__getitem__)
    result={'type':kind,'probabilities':dict(zip(labels,p))}
    if kind=='choice':
        result.update(choice=labels[best],confidence=max(0.,min(1.,(max(p)-1/len(p))/(1-1/len(p)))))
    else:
        baseline=sum(abs(i-(len(p)-1)/2) for i in range(len(p)))/len(p)
        confidence=max(0.,1-sum(v*abs(i-best) for i,v in enumerate(p))/baseline)
        result.update(score=sum(i*v for i,v in enumerate(p)),confidence=confidence,
                      legend={str(i):v if isinstance(v,str) else json.dumps(v,ensure_ascii=False) for i,v in enumerate(question['criteria'])})
    return result
