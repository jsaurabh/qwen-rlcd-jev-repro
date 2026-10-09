"""Loopback-only System One endpoint for the frozen LoRA checkpoint.
No benchmark labels or benchmark code are loaded into this process.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json, time, hashlib, traceback, os
import torch
from autojev.model import options, answer
from eval_autojev27b_vs_jev import load_model

ARTIFACT=Path(os.environ.get('DECISION_ARTIFACT', str(Path(__file__).resolve().parent)))
MODEL='our-noncausal-27b-step1533'
MAX_TOKENS=8192
BATCH_SIZE=4
model,config=load_model(ARTIFACT)
provenance={'model':MODEL,'artifact':str(ARTIFACT),'saved_config':config,
 'context_limit_per_question':MAX_TOKENS,'question_batch_size':BATCH_SIZE,
 'temperature_policy':'saved training calibration, no benchmark fitting',
 'prompt_policy':'unchanged autojev decision_messages for every benchmark',
 'server_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'capacity_policy':'255 options; 8192 tokens per complete question branch; no truncation',
 'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
Path(os.environ.get('DECISION_PROVENANCE', '/tmp/qwen-decision-provenance.json')).write_text(json.dumps(provenance,indent=2))
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,status,body):
  raw=json.dumps(body,allow_nan=False).encode();self.send_response(status)
  self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  self.reply(200,provenance if self.path=='/health' else {'error':'not found'})
 def do_POST(self):
  if self.path!='/v1/systemone':return self.reply(404,{'error':'not found'})
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<n<=32*1024*1024:raise ValueError('Request size capacity limit is 32 MiB')
   body=json.loads(self.rfile.read(n));questions=body['questions'];state=body['state']
   if body['model']!=MODEL:raise ValueError('Unknown model')
   rows=[]
   for q in questions.values():
    if q['type'] not in ('choice','noul'):raise ValueError('Unsupported question type')
    count=len(options(q)[0])
    if not 1<=count<=255:raise ValueError('maximum options per choice is 255')
    rows.append({'state':state,'question':q})
   probabilities=[];tokens=0
   with torch.inference_mode():
    for start in range(0,len(rows),BATCH_SIZE):
     batch=model.prepare(rows[start:start+BATCH_SIZE],max_length=MAX_TOKENS)
     values=(model(batch)/model.temperature).softmax(-1).cpu().tolist()
     probabilities.extend(p[:n] for p,n in zip(values,batch.counts,strict=True));tokens+=batch.input_tokens
   response={'model':MODEL,'answers':{k:answer(q,p) for (k,q),p in zip(questions.items(),probabilities,strict=True)},'usage':{'input_tokens':tokens,'output_tokens':0}}
   self.reply(200,response)
  except ValueError as exc:
   msg=str(exc)
   if 'token limit' in msg:msg='maximum context length: '+msg
   self.reply(422,{'error':msg})
  except Exception as exc:
   traceback.print_exc();self.reply(500,{'error':type(exc).__name__+': '+str(exc)})
   if isinstance(exc,torch.OutOfMemoryError) or 'device-side assert' in str(exc):
    raise SystemExit('Stop on device error; do not reuse a corrupted or exhausted GPU context')
print(json.dumps({'phase':'serving','model':MODEL,'port':8765,'max_tokens':MAX_TOKENS}),flush=True)
HTTPServer(('127.0.0.1',8765),Handler).serve_forever()
