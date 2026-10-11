"""
Aggregate the trained-reconstruction results into Table 3 of the paper.

Each CSV row is one (case, basis, K, seed, region, field) evaluation of a network
trained to 500,000 epochs: global relative L2 error (%) and the floor of the
basis it was trained with. Table 3 reports mean +/- standard deviation over the
three seeds (42, 123, 456) with the floor in brackets.

Usage:  python src/aggregate_trained.py [--csv results/trained_pod.csv results/trained_hb_dmd.csv]
"""
import argparse
import collections
import csv
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('--csv', nargs='+', default=['results/trained_pod.csv', 'results/trained_hb_dmd.csv'])
ap.add_argument('--out', default='results/table3.json')
a = ap.parse_args()

err, flo = collections.defaultdict(list), {}
for fn in a.csv:
    for r in csv.DictReader(open(fn)):
        k = (r['case'], r['basis'], int(r['K']), r['region'], r['field'])
        err[k].append(float(r['global_L2']))
        flo[k] = float(r['floor'])

COLS = [('interface', 'p'), ('all-fluid', 'u'), ('all-fluid', 'v'), ('all-fluid', 'p')]
ORDER = {'fsi3': [('pod', 6), ('hb', 7), ('dmd', 7), ('hb', 13), ('dmd', 13)],
         'fsi2': [('pod', 6), ('pod', 8), ('hb', 7), ('dmd', 7), ('hb', 13), ('dmd', 13)]}
table = {}
print('| Case | Basis, K | Interface p | Field u | Field v | Field p |')
print('|---|---|---|---|---|---|')
for case, rows in ORDER.items():
    for j, (b, K) in enumerate(rows):
        cells = []
        for reg, fld in COLS:
            g = err[(case, b, K, reg, fld)]
            assert len(g) == 3, f'{case} {b} K={K} {reg} {fld}: {len(g)} seeds'
            cells.append((st.mean(g), st.stdev(g), flo[(case, b, K, reg, fld)]))
        table[f'{case}_{b}_{K}'] = cells
        txt = ' | '.join(f'{m:.2f} ± {s:.2f} [{f:.2f}]' for m, s, f in cells)
        print(f"| {case.upper() if j == 0 else ''} | {b.upper()}, {K} | {txt} |")
json.dump(table, open(a.out, 'w'), indent=1)
print(f'\nwrote {a.out}')
