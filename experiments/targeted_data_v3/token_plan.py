"""Deterministic compute-budget plans; classification loss is averaged by decision."""
import random

def make_plan(rows, budget, seed):
    if budget <= 0 or not rows:
        raise ValueError('Need nonempty data and positive budget')
    if any(not isinstance(r['input_token_count'], int) or r['input_token_count'] <= 0 for r in rows):
        raise ValueError('Token lengths must be positive integers')
    order=list(range(len(rows)))
    random.Random(seed).shuffle(order)
    plan=[]; tokens=0
    # One pass only: no duplicates introduced to fill a budget.
    for i in order:
        n=rows[i]['input_token_count']
        if tokens+n>budget:
            break
        plan.append(i);tokens+=n
    if not plan:
        raise ValueError('Budget smaller than first example')
    if len(plan)==len(rows) and budget-tokens>=max(r['input_token_count'] for r in rows):
        raise ValueError('Not enough unique data for requested budget')
    return plan,tokens

def updates(plan, micro_batch, grad_accum):
    if min(micro_batch,grad_accum)<1:raise ValueError('Positive batch sizes required')
    size=micro_batch*grad_accum
    for start in range(0,len(plan),size):
        group=plan[start:start+size]
        yield [group[j:j+micro_batch] for j in range(0,len(group),micro_batch)]

def learning_rate(tokens_seen,budget,peak,warmup_fraction=.05):
    progress=tokens_seen/budget
    if progress<warmup_fraction:return peak*max(progress/warmup_fraction,.01)
    return peak*max(0.,(1-progress)/(1-warmup_fraction))
