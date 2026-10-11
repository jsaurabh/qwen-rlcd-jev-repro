import unittest,itertools,torch
from policy_objective import leave_one_out,policy_loss
class GradientTests(unittest.TestCase):
 def test_two_step_terminal_reward_estimator(self):
  # Exhaustively average independent two-trajectory RLOO batches in a two-step MDP.
  x=torch.tensor([[.2,-.3],[.1,.5]],dtype=torch.float64,requires_grad=True)
  lp=x.log_softmax(-1);p=lp.exp();paths=list(itertools.product(range(2),repeat=2))
  exact=-(p[0,1]*p[1,1]);expected=torch.autograd.grad(exact,x,retain_graph=True)[0]
  estimate=torch.zeros_like(x)
  for a,b in itertools.product(paths,repeat=2):
   weight=(p[0,a[0]]*p[1,a[1]]*p[0,b[0]]*p[1,b[1]]).detach()
   adv=leave_one_out([float(a==(1,1)),float(b==(1,1))],2)
   logs=torch.stack([lp[0,a[0]],lp[1,a[1]],lp[0,b[0]],lp[1,b[1]]])
   loss=policy_loss(logs,torch.tensor([adv[0],adv[0],adv[1],adv[1]],dtype=x.dtype),2)
   estimate+=weight*torch.autograd.grad(loss,x,retain_graph=True)[0]
  torch.testing.assert_close(estimate,expected,rtol=1e-10,atol=1e-10)
 def test_baseline_no_grad(self):
  lp=torch.tensor([-.3,-.2],requires_grad=True);a=torch.tensor([1.,-1.],requires_grad=True)
  policy_loss(lp,a,2).backward();self.assertIsNone(a.grad)
if __name__=='__main__':unittest.main()
