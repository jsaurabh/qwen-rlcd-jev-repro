"""Evaluate a saved local MLX schema adapter on Nimble-format public records."""
import argparse, json, sys
from pathlib import Path
from mlx_lm import load

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference-nimble'))
from nimble.training.schema_data import as_scoring, encode_scoring
from nimble_1_5b_mlx import dump, evaluate, sha

def encode(rows,tokenizer,max_tokens,shuffle_seed=None):
    out=[]
    for raw in rows:
        item=encode_scoring(as_scoring(raw,False),tokenizer,max_tokens,shuffle_seed)
        out.append({'id':raw['id'],'pair':raw['family'],'family':raw.get('source_family',raw['family']),'domain':raw['domain'],
                    'kind':raw['input']['questions']['decision']['type'],'input_ids':item['prompt_token_ids'],
                    'candidate_ids':item['candidate_token_ids'],'label':item['target_index'],
                    'choices':list(item['code_to_choice'].values())})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--adapter',type=Path)
    ap.add_argument('--model',default='models/qwen-1.5b-4bit');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-tokens',type=int,default=1152);ap.add_argument('--shuffle-seed',type=int);args=ap.parse_args()
    if args.out.exists():ap.error('output exists; choose a new path')
    rows=[json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
    if not rows or any('teacher' in r for r in rows):raise ValueError('Requires nonempty teacher-free public records')
    model,tok=load(str(ROOT/args.model),adapter_path=str(args.adapter) if args.adapter else None);encoded=encode(rows,tok,args.max_tokens,args.shuffle_seed)
    records,summary,seconds=evaluate(model,encoded)
    dump(args.out,{'dataset':str(args.data),'dataset_sha256':sha(args.data),'adapter':str(args.adapter) if args.adapter else None,
                   'shuffle_seed':args.shuffle_seed,'max_tokens':args.max_tokens,'seconds':seconds,'summary':summary,'rows':records})
    print(json.dumps({'output':str(args.out),'summary':summary},indent=2))

if __name__=='__main__':main()
