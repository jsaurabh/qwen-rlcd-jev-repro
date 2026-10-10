import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"targeted_data_v3"))
from token_plan import make_plan,updates
class FullEpochTest(unittest.TestCase):
 def test_every_row_once_and_partial_update(self):
  rows=[{"input_token_count":1+i%480} for i in range(92809)]
  budget=sum(r["input_token_count"] for r in rows)
  a,t=make_plan(rows,budget,20261010);b,u=make_plan(rows,budget,20261010)
  self.assertEqual(a,b);self.assertEqual(t,budget);self.assertEqual(len(set(a)),len(rows))
  groups=list(updates(a,4,8));self.assertEqual(len(groups),2901)
  self.assertEqual(sum(map(len,groups[-1])),9)
if __name__=="__main__":unittest.main()
