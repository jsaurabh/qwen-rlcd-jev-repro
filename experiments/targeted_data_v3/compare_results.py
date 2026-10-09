"""Compare aggregate reports from the fixed development fold (not leaderboard scores)."""
import argparse
import json
from pathlib import Path


def compare(control, targeted):
    if control['development']['count'] != targeted['development']['count']:
        raise ValueError('Different evaluation counts')
    if set(control['by_suite']) != set(targeted['by_suite']):
        raise ValueError('Different evaluation suites')
    metrics = ['accuracy', 'nll', 'brier', 'ece']
    def differences(a, b):
        return {key: {'control': a[key], 'targeted': b[key], 'delta': b[key] - a[key]}
                for key in metrics}
    suites = {}
    for name, a in control['by_suite'].items():
        b = targeted['by_suite'][name]
        if a['count'] != b['count']:
            raise ValueError(f'Different suite counts: {name}')
        suites[name] = {'count': a['count'], **differences(a['calibrated'], b['calibrated'])}
    return {
        'evaluation': 'Shared development set; separately fitted calibration temperature per arm',
        'limitations': 'One seed, short compute-matched pilot. Aggregate reports cannot establish paired statistical significance. These are not Decision Index scores.',
        'token_budget': {'control': control['tokens_seen'], 'targeted': targeted['tokens_seen'],
                         'difference': targeted['tokens_seen'] - control['tokens_seen']},
        'overall': differences(control['development'], targeted['development']),
        'by_suite': suites,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('targeted', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.control.read_text()), json.loads(args.targeted.read_text())), indent=2))
