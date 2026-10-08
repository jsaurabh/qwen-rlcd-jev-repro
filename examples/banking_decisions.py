"""Run from the repository root: python -m examples.banking_decisions --provider local."""
import argparse
import json
import os
from decision_api import DecisionsClient, SystemOneClient

STATE = 'I see two card charges I do not recognize. My card is still with me. Please help stop further charges.'
QUESTIONS = [
    {'type':'predicate','name':'review','instructions':'Does the customer report suspected unauthorized activity requiring human review?'},
    {'type':'choice','name':'team','instructions':'Which team should handle this case?', 'choices':[
        {'value':'payments','description':'Known transfers, charges, or payment status.'},
        {'value':'account_support','description':'Account access or servicing.'},
        {'value':'fraud_review','description':'Suspected unauthorized activity.'}]},
    {'type':'score','name':'urgency','instructions':'Rate the urgency of this case.', 'levels':[
        {'label':'Routine','description':'No immediate disruption or loss.'},
        {'label':'Prompt','description':'Access or payment disruption.'},
        {'label':'Immediate','description':'Suspected ongoing unauthorized activity or imminent loss.'}]},
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=['local','openai','jev'],default='local')
    p.add_argument('--base-url');p.add_argument('--model')
    args=p.parse_args()
    defaults={'local':('http://127.0.0.1:8000/v1','our-autojev','DECISION_API_KEY'),
              'openai':('https://api.openai.com/v1','gpt-6-luna','OPENAI_API_KEY'),
              'jev':('https://api.typesafe.ai/v1','jev-latest','TYPESAFE_API_KEY')}
    url,model,keyvar=defaults[args.provider]
    key=os.environ.get(keyvar)
    if args.provider!='local' and not key:p.error(f'Set {keyvar} in your environment')
    cls=SystemOneClient if args.provider=='jev' else DecisionsClient
    client=cls(base_url=args.base_url or url,api_key=key)
    method=client.from_decisions if args.provider=='jev' else client.create
    print(json.dumps(method(model=args.model or model,input=STATE,questions=QUESTIONS),indent=2))


if __name__=='__main__':main()
