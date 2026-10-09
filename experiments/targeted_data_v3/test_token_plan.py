import unittest
from token_plan import make_plan, updates, learning_rate
class Tests(unittest.TestCase):
 def test_budget_and_resume(self):
  rows=[{'input_token_count':n} for n in range(1,481)]*20
  plan,t=make_plan(rows,100000,12)
  self.assertEqual((plan,t),make_plan(rows,100000,12))
  self.assertEqual(len(plan),len(set(plan)))
  self.assertLessEqual(t,100000);self.assertLess(100000-t,480)
  groups=list(updates(plan,4,8));cursor=sum(len(b) for g in groups[:3] for b in g)
  self.assertEqual([i for g in groups[3:] for b in g for i in b],plan[cursor:])
 def test_partial_gradient_weight(self):
  # Each microbatch mean must be weighted by its decision count, including tail.
  group=list(updates(list(range(11)),4,8))[0]
  losses=list(range(11));weighted=sum(sum(losses[i] for i in b)/len(b)*len(b)/11 for b in group)
  self.assertAlmostEqual(weighted,sum(losses)/11)
 def test_schedule(self):
  self.assertAlmostEqual(learning_rate(50000,1000000,1),1)
  self.assertAlmostEqual(learning_rate(1000000,1000000,1),0)
if __name__=='__main__':unittest.main()
