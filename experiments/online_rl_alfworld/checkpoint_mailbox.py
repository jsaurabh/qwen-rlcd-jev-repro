"""Block until a separate credential-holding controller verifies a Drive upload."""
import json,time,os
from pathlib import Path
class CheckpointMailbox:
    def __init__(self, root, timeout=1800):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.timeout=timeout
    def upload_checkpoint(self,path):
        path=Path(path).resolve()
        if not (path/'COMPLETE.json').is_file():raise ValueError('Incomplete checkpoint')
        request=self.root/(path.name+'.request.json')
        receipt=self.root/(path.name+'.receipt.json')
        if request.exists() or receipt.exists():raise FileExistsError('Duplicate checkpoint upload')
        partial=request.with_suffix('.tmp')
        partial.write_text(json.dumps({'path':str(path)}));os.replace(partial,request)
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            error=self.root/(path.name+'.error.json')
            if error.exists():raise RuntimeError('Checkpoint controller upload failed; see root log')
            if receipt.exists():return json.loads(receipt.read_text())
            time.sleep(2)
        raise TimeoutError('Drive verification did not finish; training stopped')
