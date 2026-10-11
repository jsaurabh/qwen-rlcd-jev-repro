from pathlib import Path
import sys,subprocess,time,json,os,shutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
from drive_checkpoint_store import DriveStore,MARKER,hashes
R=Path('/content/online-rl-v3');W=R/'work';C=R/'code';cred=Path('/content/.checkpoint-auth/credentials.json')
ROOT=os.environ['DRIVE_ROOT_FOLDER']
INITIAL_FOLDER=os.environ['INITIAL_CHECKPOINT_FOLDER']
RETENTION_FOLDER=os.environ['RETENTION_DATA_FOLDER']
RESUME=Path(sys.argv[1]) if len(sys.argv)>1 else None
def verify_local(store,folder,path):
 markers=[x for x in store.children(folder) if x['name']==MARKER]
 assert len(markers)==1
 for name,meta in store.read_json(markers[0]['id'])['files'].items():
  p=path/name
  assert p.resolve().is_relative_to(path.resolve())
  assert p.stat().st_size==meta['size'] and hashes(p)[1]==meta['sha256'],name
def status(phase,**kw):
 d={'phase':phase,'time':time.time(),**kw};(R/'status.json').write_text(json.dumps(d));print(json.dumps(d),flush=True)
def own(path):
 os.chown(path,65534,65534)
 for p in path.rglob('*'):os.chown(p,65534,65534)
def lower():os.setgroups([]);os.setgid(65534);os.setuid(65534)
def upload(path,folder):
 for attempt in range(6):
  try:return DriveStore(str(cred),folder).upload_checkpoint(path)
  except (TimeoutError,ConnectionError,OSError) as e:
   if attempt==5:raise
   status('upload_retry',checkpoint=path.name,attempt=attempt+1,error_type=type(e).__name__);time.sleep(min(60,5*2**attempt))
try:
 assert json.loads((R/'setup-status.json').read_text())['phase']=='ready_for_restore'
 cred.chmod(0o600);store=DriveStore(str(cred),ROOT)
 folder=store.folder(os.environ.get('EXPERIMENT_NAME','online-rl-alfworld-v3'));(R/'experiment.json').write_text(json.dumps({'folder_id':folder}))
 if RESUME:
  status('verifying_recovery',drive_folder=folder)
  assert RESUME.resolve()==(W/'policy/checkpoint-00000').resolve()
  assert json.loads((RESUME/'COMPLETE.json').read_text())['step']==0
  verify_local(store,INITIAL_FOLDER,W/'initial')
  verify_local(store,RETENTION_FOLDER,W/'ready')
  receipt=json.loads((W/'queue/checkpoint-00000.receipt.json').read_text())
  verify_local(store,receipt['folder_id'],RESUME)
  for name in ['baseline.json','split_manifest.json']:
   assert (W/'policy'/name).read_bytes()==(RESUME/name).read_bytes()
 else:
  status('restoring_initial_sft',drive_folder=folder)
  store.restore(INITIAL_FOLDER,W/'initial')
  status('restoring_retention_data');store.restore(RETENTION_FOLDER,W/'ready')
 snapshot=R/('source-snapshot-recovery-'+str(int(time.time())) if RESUME else 'source-snapshot');snapshot.mkdir();shutil.copytree(C,snapshot/'code')
 for n in ['model-freeze.txt','env-freeze.txt']:shutil.copyfile(R/n,snapshot/n)
 (snapshot/'COMPLETE.json').write_text('{}');upload(snapshot,folder)
 q=W/'queue';q.mkdir(exist_ok=bool(RESUME));out=W/'policy';out.mkdir(exist_ok=bool(RESUME));own(W)
 env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(W),'HF_HOME':str(W/'hf-cache'),'PYTHONPATH':str(C),'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH',''),'TOKENIZERS_PARALLELISM':'false','LANG':'C.UTF-8'}
 with (R/'tests.log').open('w') as f:
  subprocess.run([sys.executable,'-m','unittest','discover','-s',str(C),'-p','test_*.py'],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=W,preexec_fn=lower,check=True)
 status('tests_passed_loading_model')
 command=[sys.executable,'-u',str(C/'train_online.py'),'--initial',str(W/'initial'),'--out',str(out),'--data',str(W/'alfworld'),'--env-python',str(R/'env-venv/bin/python'),'--env-worker',str(C/'env_worker.py'),'--checkpoint-spool',str(q),'--retention-data',str(W/'ready')]
 if RESUME:command.extend(['--resume',str(RESUME)])
 (R/'command.json').write_text(json.dumps(command))
 with (R/'train.log').open('a' if RESUME else 'w') as log:
  proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=W,preexec_fn=lower,start_new_session=True)
  status('running',pid=proc.pid,drive_folder=folder)
  try:
   while proc.poll() is None:
    for request in sorted(q.glob('*.request.json')):
     receipt=request.with_name(request.name.replace('.request.json','.receipt.json'))
     if receipt.exists():continue
     source=Path(json.loads(request.read_text())['path'])
     assert source.resolve().parent==out.resolve() and not source.is_symlink()
     assert not any(p.is_symlink() for p in source.rglob('*'))
     result=upload(source,folder);tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(receipt)
     status('checkpoint_verified',checkpoint=source.name,receipt=result)
     old=sorted(p for p in out.glob('checkpoint-*') if p.is_dir())
     for p in old[:-2]:
      if (q/(p.name+'.receipt.json')).exists():shutil.rmtree(p)
    time.sleep(2)
   if proc.returncode:raise RuntimeError(f'Trainer exited {proc.returncode}')
  except BaseException:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
   raise
 metadata=R/'final-metadata';metadata.mkdir()
 for p in out.glob('*.json*'):shutil.copyfile(p,metadata/p.name)
 shutil.copyfile(R/'train.log',metadata/'run.log');(metadata/'COMPLETE.json').write_text('{}')
 receipt=upload(metadata,folder);status('complete',drive_folder=folder,receipt=receipt)
except BaseException as e:
 status('failed',error_type=type(e).__name__,error=str(e));raise
finally:cred.unlink(missing_ok=True)
