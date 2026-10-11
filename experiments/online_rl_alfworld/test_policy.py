import unittest
from policy_objective import leave_one_out,trajectory_return,matching_rollout_batches
class ObjectiveTests(unittest.TestCase):
 def test_loo_excludes_self(self):
  self.assertEqual(leave_one_out([1,0,0,0],4),[1,-1/3,-1/3,-1/3])
  self.assertEqual(leave_one_out([1,1,1,1],4),[0,0,0,0])
 def test_independent_groups(self):
  self.assertEqual(leave_one_out([1,0,2,4],2),[1,-1,-2,2])
 def test_terminal_objective(self):
  self.assertAlmostEqual(trajectory_return(True,10,0),.98)
  self.assertAlmostEqual(trajectory_return(False,50,0),-.1)
 def test_bad_groups(self):
  with self.assertRaises(ValueError):leave_one_out([1,0,1],2)
 def test_recompute_preserves_rollout_batches(self):
  trajectories=[{'transitions':['a0','a1','a2']},{'transitions':['b0']},{'transitions':['c0','c1']}]
  batches=list(matching_rollout_batches(trajectories,[1,2,3],2))
  self.assertEqual(batches,[[('a0',1),('b0',2)],[('c0',3)],[('a1',1),('c1',3)],[('a2',1)]])
if __name__=='__main__':unittest.main()
