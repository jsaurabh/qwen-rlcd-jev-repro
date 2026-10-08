"""Run unchanged upstream code against the local 0.5B checkpoint."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'upstream'))
from mlx_lm import load
from core import engine_mlx
from core.schema import StructuredSchema
model, tokenizer = load('models/qwen-0.5b-4bit')

# dependency injection avoids downloading upstream's 1.5B default; no source edits.
engine_mlx._model, engine_mlx._tokenizer = model, tokenizer
schema = StructuredSchema({
    'department': {'type':'enum', 'choices':['billing','technical','security'], 'description':'Which team handles the issue: billing, technical, or security?'},
    'urgency': {'type':'enum','choices':['low','medium','high'],'description':'How urgent is the incident?'},
    'escalate': {'type':'boolean','description':'Should an outage be escalated?'}
})

result=engine_mlx.run_parallel_generation('All customers are blocked by a production service outage.',schema)

assert result['schema_match'] and result['is_valid_json']
assert set(result['parsed_json']) == {'department','urgency','escalate'}

Path('runs/upstream_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
