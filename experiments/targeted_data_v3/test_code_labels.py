"""Verify the owned synthetic templates against Python, without running the builder."""
import ast
from pathlib import Path
import random
import unittest

class CodeLabelTests(unittest.TestCase):
    def test_gold_formulas_match_expression_semantics(self):
        tree = ast.parse(Path(__file__).with_name('build_candidates.py').read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'expressions' for t in node.targets))
        module = ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[]))
        builtins = {'sum': sum, 'range': range, 'len': len, 'max': max, 'min': min, 'abs': abs}
        env = {'__builtins__': builtins}
        exec(compile(module, 'owned-code-templates', 'exec'), env)
        rng = random.Random(20261009)
        for expression, gold in env['expressions']:
            for _ in range(100):
                a, b, c = rng.randint(1, 30), rng.randint(1, 30), rng.randint(1, 9)
                actual = eval(expression, {'__builtins__': builtins, 'a': a, 'b': b, 'c': c})
                self.assertEqual(gold(a, b, c), actual, (expression, a, b, c))

if __name__ == '__main__':
    unittest.main()
