import unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from reference_cache import reference_cache
class Fake:
 def __init__(self):self.calls=0
 def eval(self):return self
 def train(self):return self
 def prepare(self,rows,max_length):return SimpleNamespace(inputs={'attention_mask':torch.ones(len(rows),3)})
 def __call__(self,batch):self.calls+=1;return torch.zeros(len(batch.inputs['attention_mask']),255)
class Tests(unittest.TestCase):
 def test_cache_reuse_backward_and_identity(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'ref.pt';m=Fake();rows=[{'input_token_count':3}]*3;identity={'max_length':512,'plan':'abc'}
   ref,meta=reference_cache(m,rows,[0,1,2],p,identity,2)
   self.assertEqual(m.calls,2)
   x=torch.randn(3,255,requires_grad=True);(x*ref).sum().backward();self.assertIsNotNone(x.grad)
   ref2,_=reference_cache(m,rows,[0,1,2],p,identity,2);torch.testing.assert_close(ref,ref2);self.assertEqual(m.calls,2)
   with self.assertRaises(ValueError):reference_cache(m,rows,[0,1,2],p,{**identity,'plan':'other'},2)
 def test_resume_cannot_create_reference(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):reference_cache(Fake(),[],[],Path(t)/'ref.pt',{},1,allow_create=False)
if __name__=='__main__':unittest.main()
