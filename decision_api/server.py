"""Single-process research server for saved AutoJev LoRA artifacts."""
import argparse
import hmac
import json
import logging
import math
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from .protocol import answer, decisions_to_systemone, systemone_to_decisions, validate_systemone


class AutoJevBackend:
    def __init__(self, artifact, batch_size=4):
        # Lazy imports keep client/tests usable without CUDA or model packages.
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from autojev.model import DecisionModel
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'colab'))
        from autojev_lora_train import enable_noncausal_full_attention
        artifact=Path(artifact)
        self.config=json.loads((artifact/'decision_config.json').read_text())
        self.temperature=float(self.config['temperature'])
        if not math.isfinite(self.temperature) or self.temperature<=0:raise ValueError('Invalid saved temperature')
        if batch_size<1:raise ValueError('batch_size must be positive')
        self.batch_size=batch_size
        self.model=DecisionModel(train=False,base_model=self.config['base_model'],revision=self.config['revision'])
        mode=self.config.get('attention_mode','causal')
        if mode=='noncausal_full_attention':enable_noncausal_full_attention(self.model.backbone.language_model)
        elif mode!='causal':raise ValueError('Unsupported saved attention mode')
        self.model.backbone=PeftModel.from_pretrained(self.model.backbone,artifact/'adapter').eval()
        self.model.readout.load_state_dict(load_file(str(artifact/'readout.safetensors')))
        self.model.processor=self.model.processor.from_pretrained(str(artifact/'processor'))
        self.model.processor.tokenizer.padding_side='left'
        self.model.eval()

    def predict(self, rows):
        import torch
        results, tokens=[],0
        with torch.inference_mode():
            for i in range(0,len(rows),self.batch_size):
                batch=self.model.prepare(rows[i:i+self.batch_size],max_length=self.config['max_length'])
                p=(self.model(batch)/self.temperature).softmax(-1).cpu().tolist()
                results.extend(values[:n] for values,n in zip(p,batch.counts,strict=True))
                tokens+=batch.input_tokens
        return results,tokens


def create_app(backend, *, model_name='our-autojev', api_key=None):
    app=FastAPI(title='Decision model research API',version='0.1.0')
    lock=threading.Lock()
    def evaluate(body):
        validate_systemone(body)
        if body['model']!=model_name:raise HTTPException(404, 'Unknown model')
        rows=[{'state':body['state'],'question':q} for q in body['questions'].values()]
        # Model inference is serialized so concurrent requests cannot exhaust VRAM.
        with lock:distributions,tokens=backend.predict(rows)
        if len(distributions)!=len(rows):raise RuntimeError('Backend answer count mismatch')
        usage={'input_tokens':tokens,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},
               'output_tokens':0,'output_tokens_details':{'reasoning_tokens':0},'total_tokens':tokens}
        return {'model':model_name,'answers':{k:answer(q,p) for (k,q),p in zip(body['questions'].items(),distributions,strict=True)},'usage':usage}

    async def payload(request):
        if api_key and not hmac.compare_digest(request.headers.get('authorization',''),f'Bearer {api_key}'):
            raise HTTPException(401,'Invalid API key')
        raw=bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw)>1024*1024:raise HTTPException(413,'Request exceeds 1 MiB')
        try:
            return json.loads(raw,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number')))
        except (ValueError,UnicodeDecodeError):raise HTTPException(422,'Invalid JSON') from None

    async def run(body, decisions=False):
        try:
            if decisions:body,mapping=decisions_to_systemone(body)
            result=await run_in_threadpool(evaluate,body)
            return systemone_to_decisions(result,mapping) if decisions else result
        except ValueError as exc:raise HTTPException(422,str(exc)) from None
        except HTTPException:raise
        except Exception:
            logging.getLogger(__name__).exception('Decision inference failed')
            raise HTTPException(500,'Model inference failed') from None

    @app.get('/health')
    def health():return {'status':'ready','model':model_name,'modalities':['text'], 'confidence':'typesafe-formulas', 'shared_prefill':False}

    @app.post('/v1/decisions')
    async def decisions(request:Request):return await run(await payload(request),True)

    @app.post('/v1/systemone')
    async def systemone(request:Request):return await run(await payload(request))

    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--model-name',default='our-autojev')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000)
    p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args()
    key=os.environ.get('DECISION_API_KEY')
    if args.host not in ('127.0.0.1','localhost','::1') and not key:
        p.error('Set DECISION_API_KEY before binding a non-loopback address')
    backend=AutoJevBackend(args.artifact,args.batch_size)
    import uvicorn
    uvicorn.run(create_app(backend,model_name=args.model_name,api_key=key),host=args.host,port=args.port,workers=1)


if __name__=='__main__':main()
