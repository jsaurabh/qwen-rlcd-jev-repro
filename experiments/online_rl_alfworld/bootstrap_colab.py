from pathlib import Path
import subprocess
r=Path('/content/online-rl-v3');r.mkdir(exist_ok=True)
p=Path('/content/.checkpoint-auth');p.mkdir(mode=0o700,exist_ok=True);p.chmod(0o700)
script=r/'setup.py'
script.write_text('''from pathlib import Path
import subprocess,sys,os,json,time,zipfile,urllib.request,shutil
R=Path('/content/online-rl-v3');W=R/'work';W.mkdir(exist_ok=True)
def status(phase):
 (R/'setup-status.json').write_text(json.dumps({'phase':phase,'time':time.time()}));print(phase,flush=True)
try:
 status('installing_model_dependencies')
 subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
 subprocess.run([sys.executable,'-m','pip','install','-r','/content/online-rl-requirements.txt','google-api-python-client','google-auth-httplib2','uv'],check=True)
 status('installing_cpu_environment')
 # Keep the managed interpreter accessible to the unprivileged simulator worker.
 subprocess.run(['uv','venv','--python','3.11',str(R/'env-venv')],check=True,env={**os.environ,'UV_PYTHON_INSTALL_DIR':str(R/'python-runtime')})
 py=str(R/'env-venv/bin/python')
 subprocess.run(['uv','pip','install','--python',py,'git+https://github.com/alfworld/alfworld.git@aaba6870f86c5be6a08a491f32a50b906227bc3e','textworld==1.7.0','termcolor','tqdm','numpy','requests'],check=True)
 status('downloading_environment_data')
 data=W/'alfworld';data.mkdir(exist_ok=True)
 for url in ['https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip','https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip','https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip']:
  dst=data/url.split('/')[-1];urllib.request.urlretrieve(url,dst)
  with zipfile.ZipFile(dst) as z:
   for name in z.namelist():
    if not (data/name).resolve().is_relative_to(data.resolve()):raise ValueError('Unsafe archive path')
   z.extractall(data)
  dst.unlink()
 (R/'env-freeze.txt').write_bytes(subprocess.check_output(['uv','pip','freeze','--python',py]))
 (R/'model-freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
 status('ready_for_restore')
except BaseException as e:
 status('failed_'+type(e).__name__);raise
''')
with (r/'setup.log').open('w') as f:
 p=subprocess.Popen(['/usr/bin/python3','-u',str(script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('setup_pid',p.pid)
