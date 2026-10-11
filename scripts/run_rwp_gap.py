"""RWP and Cascade on the 180 adversarial instances of RQ1 (Table 1).
Run from the repository root:  python scripts/run_rwp_gap.py"""
import sys, csv
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import *
from mccf_ilp import exact_ilp
from run_repair_ratio import recompute_whenever_possible
rows = []
for d in (1, 2, 3):
    for n in (50, 100, 200):
        for a in (3, 5):
            for seed in range(1, 11):
                g = generate_family_cfg(n, a, 'ADVERSARIAL', seed, trap_depth=d); f = {0}
                p, st = exact_ilp(g, f, time_limit=300)
                rows.append(dict(trap_depth=d, n=n, actors=a, seed=seed, optimal=round(total_cost(g, p), 6),
                                 rwp=round(total_cost(g, recompute_whenever_possible(g, f)), 6),
                                 cascade=round(total_cost(g, cascade_delete(g, f)), 6)))
with open('results/results_rwp_gap.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
