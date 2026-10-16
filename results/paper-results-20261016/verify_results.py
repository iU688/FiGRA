"""Verify the released result files with the Python standard library."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, math, statistics

root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST_SHA256.json').read_text('utf8'))
for name,digest in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
def rows(name):
    with (root/name).open(encoding='utf8',newline='') as f: return list(csv.DictReader(f))
def same(a,b): assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9),(a,b)
for table in (1,2):
    values=defaultdict(list)
    for r in rows(f'Table_{table}_seed_values.csv'):
        values[r['method'],r['dataset'],r['metric']].append(float(r['value_percent']))
    for r in rows(f'Table_{table}_summary.csv'):
        for target,prefix in [('humaneva','H'),('mupots','M')]:
            for metric in ('PCK','AUC'): same(statistics.mean(values[r['method'],target,metric]),r[prefix+'_'+metric])
values=defaultdict(list)
for r in rows('Figure_4_seed_values.csv'): values[r['dataset'],r['group'],r['metric']].append(float(r['gain_percentage_points']))
for r in rows('Figure_4_gains.csv'):
    v=values[r['dataset'],r['group'],r['metric']]
    assert len(v)==3
    same(statistics.mean(v),r['gain_percentage_points']);same(statistics.stdev(v),r['sample_sd_percentage_points'])
values=defaultdict(list)
for r in rows('Figure_7_seed_values.csv'):
    values[r['method'],r['dataset'],r['trainable_parameters'],r['metric']].append(float(r['value_percent']))
for r in rows('Figure_7_budget.csv'):
    v=values[r['method'],r['dataset'],r['trainable_parameters'],r['metric']]
    assert len(v)==3
    same(statistics.mean(v),r['mean_percent']);same(statistics.stdev(v),r['sample_sd_percent'])
assert len(rows('Figure_6_coordinate_jitter.csv'))==54
assert len(rows('Figure_5_case_index.csv'))==6
assert len(rows('Figure_8_case_index.csv'))==4
assert len(rows('Figure_5_displayed_predictions.csv'))==6*2*17
assert len(rows('Figure_8_displayed_predictions.csv'))==4*2*17
print('PASS: file hashes, table means, plotted error bars, and qualitative case counts.')
