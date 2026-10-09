import unittest,tempfile,threading,json
from pathlib import Path
from checkpoint_mailbox import CheckpointMailbox
class Tests(unittest.TestCase):
 def test_waits_for_verified_receipt_and_rejects_reuse(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cp=root/'checkpoint-00001';cp.mkdir();(cp/'COMPLETE.json').write_text('{}')
   box=CheckpointMailbox(root/'queue',timeout=5)
   def controller():
    import time
    request=box.root/'checkpoint-00001.request.json'
    while not request.exists():time.sleep(.01)
    self.assertEqual(json.loads(request.read_text())['path'],str(cp.resolve()))
    receipt=box.root/'checkpoint-00001.receipt.json';partial=receipt.with_suffix('.tmp');partial.write_text('{"verified": true}');partial.replace(receipt)
   t=threading.Thread(target=controller);t.start()
   self.assertEqual(box.upload_checkpoint(cp),{'verified':True});t.join()
   with self.assertRaises(FileExistsError):box.upload_checkpoint(cp)
 def test_refuses_incomplete(self):
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):CheckpointMailbox(Path(tmp)/'q').upload_checkpoint(tmp)
if __name__=='__main__':unittest.main()
