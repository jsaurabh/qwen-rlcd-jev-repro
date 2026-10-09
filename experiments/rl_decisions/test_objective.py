import unittest
import torch
from objective import decision_rl_loss
class Tests(unittest.TestCase):
 def test_exact_policy_gradient(self):
  x=torch.tensor([[.7,-.2,1.3]],requires_grad=True);q=torch.tensor([[0.,0.,1.]])
  ref=x.detach().log_softmax(-1);loss,_=decision_rl_loss(x,q,ref,[3],beta=0,ce_weight=0);loss.backward()
  p=x.detach().softmax(-1);expected=-p*(q-(p*q).sum(-1,keepdim=True))
  torch.testing.assert_close(x.grad,expected)
 def test_identical_reference_zero_kl_and_detached(self):
  x=torch.tensor([[.3,.8]],requires_grad=True);ref=x.detach().clone().log_softmax(-1).requires_grad_(True)
  loss,stats=decision_rl_loss(x,torch.tensor([[.4,.6]]),ref,[2]);loss.backward()
  self.assertAlmostEqual(stats['kl'].item(),0,places=6);self.assertIsNone(ref.grad)
 def test_invalid_padding_never_affects_gradient(self):
  x=torch.tensor([[.3,.8,900.]],requires_grad=True);ref=torch.tensor([[-.974077,-.474077,float('-inf')]])
  loss,_=decision_rl_loss(x,torch.tensor([[0.,1.,0.]]),ref,[2]);loss.backward()
  self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(x.grad[0,2].item(),0.)
 def test_ce_and_kl_retain_gradients(self):
  x=torch.tensor([[.4,.1]],requires_grad=True);q=torch.tensor([[.2,.8]]);ref=torch.tensor([[.7,.3]]).log()
  loss,stats=decision_rl_loss(x,q,ref,[2]);self.assertGreater(stats['kl'].item(),0)
  grad=torch.autograd.grad(loss,x)[0];eps=.001
  plus=x.detach().clone();minus=plus.clone();plus[0,0]+=eps;minus[0,0]-=eps
  numeric=(decision_rl_loss(plus,q,ref,[2])[0]-decision_rl_loss(minus,q,ref,[2])[0])/(2*eps)
  self.assertAlmostEqual(grad[0,0].item(),numeric.item(),places=4)
 def test_partial_accumulation(self):
  torch.manual_seed(4);x=torch.randn(7,3,requires_grad=True);q=torch.eye(3)[torch.arange(7)%3];ref=torch.randn(7,3).log_softmax(-1)
  full=decision_rl_loss(x,q,ref,[3]*7)[0];a=torch.autograd.grad(full,x)[0]
  partial=sum(decision_rl_loss(x[i:j],q[i:j],ref[i:j],[3]*(j-i))[0]*(j-i)/7 for i,j in [(0,4),(4,7)])
  torch.testing.assert_close(a,torch.autograd.grad(partial,x)[0])
if __name__=='__main__':unittest.main()
