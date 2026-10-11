"""Summarize paired ALFWorld task outcomes; no new model evaluation."""
import argparse
import json
import math
from pathlib import Path


def outcomes(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {row['task']: bool(row['won']) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate evaluation task in {path}')
    return result


def compare(before, after):
    if before.keys() != after.keys():
        raise ValueError('Evaluation task sets differ')
    n = len(before)
    if not n:
        raise ValueError('Empty evaluation')
    gained = sum(not before[k] and after[k] for k in before)
    lost = sum(before[k] and not after[k] for k in before)
    discordant = gained + lost
    p = min(1.0, 2 * sum(math.comb(discordant, j)
                        for j in range(min(gained, lost) + 1)) / 2**discordant)
    return {
        'tasks': n,
        'baseline_successes': sum(before.values()),
        'final_successes': sum(after.values()),
        'delta_percentage_points': 100 * (gained - lost) / n,
        'gained_tasks': gained,
        'lost_tasks': lost,
        'mcnemar_exact_two_sided_p': p,
        'note': 'Small feasibility sample; p-value is descriptive, not a transfer claim.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    summary = {}
    compact_path = args.results / 'paired-outcomes.jsonl'
    compact = [json.loads(line) for line in compact_path.read_text().splitlines()] if compact_path.exists() else None
    for split in ['valid_seen', 'valid_unseen']:
        if compact is not None:
            rows = [row for row in compact if row['split'] == split]
            before = {row['task']: bool(row['baseline_won']) for row in rows}
            after = {row['task']: bool(row['final_won']) for row in rows}
            if len(before) != len(rows):
                raise ValueError(f'Duplicate evaluation task in {split}')
        else:
            before = outcomes(args.results / f'baseline-{split}-trajectories.jsonl')
            after = outcomes(args.results / f'final-{split}-trajectories.jsonl')
        summary[split] = compare(before, after)
    baseline = json.loads((args.results / 'baseline.json').read_text())
    final = json.loads((args.results / 'final.json').read_text())
    summary['retention'] = {'baseline': baseline['retention'], 'final': final['retention']}
    summary['updates'] = final['updates']
    summary['stop_reason'] = final['stop_reason']
    summary['training_wall_seconds'] = final['training_wall_seconds']
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
