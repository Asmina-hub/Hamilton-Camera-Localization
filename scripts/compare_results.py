#!/usr/bin/env python3
"""Side-by-side table of pipeline variants run on the same test files (scripts/run_testsets.py --tag).

    python scripts/compare_results.py ~/camloc_testsets --tags ,roma --names baseline,roma_fallback

Reads <root>/results[_<tag>].csv for every tag ('' = untagged run) and prints / writes <root>/compare.md with one
row per lap x scenario: position RMSE, lateral, longitudinal, within 1 m for each variant, and the difference.
"""
import argparse, os
import pandas as pd

COLS = ['rmse', 'lateral', 'longitudinal', 'within_1m', 'coverage']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--tags', default=',roma', help='comma-separated tags, empty = the untagged run')
    ap.add_argument('--names', default=None, help='display names, same order as --tags')
    a = ap.parse_args()
    tags = a.tags.split(',')
    names = a.names.split(',') if a.names else [t or 'baseline' for t in tags]
    merged = None
    for tag, name in zip(tags, names):
        path = os.path.join(a.root, f'results{"_" + tag if tag else ""}.csv')
        d = pd.read_csv(path)
        d = d[['lap', 'scenario'] + [c for c in COLS if c in d]]
        d = d.rename(columns={c: f'{c}:{name}' for c in COLS})
        merged = d if merged is None else merged.merge(d, on=['lap', 'scenario'], how='outer')
    if len(names) == 2:
        b, v = names
        for c in ('rmse', 'lateral', 'within_1m'):
            if f'{c}:{b}' in merged and f'{c}:{v}' in merged:
                merged[f'd_{c}'] = merged[f'{c}:{v}'] - merged[f'{c}:{b}']
    merged = merged.sort_values(['scenario', 'lap'])
    try:
        md = merged.to_markdown(index=False, floatfmt='.3f')
    except ImportError:
        md = merged.to_string(index=False)
    with open(os.path.join(a.root, 'compare.md'), 'w') as f:
        f.write(md + '\n')
    print(md)
    print(f'-> {os.path.join(a.root, "compare.md")}')


if __name__ == '__main__':
    main()
