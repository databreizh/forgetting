"""Action ablation: optimal MCCF cost when unlearn or recompute is disabled.
Run from the repository root:  python scripts/run_ablation.py"""
import sys, csv, random
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import *
from mccf_ilp import exact_ilp

def disable(g, mode):
    for nd in g.nodes.values():
        if mode == 'no_unlearn': nd.unlearnable = False
        if mode == 'no_recompute': nd.recomputable = False

rows = []
for fam in ('LOCAL', 'SHARED', 'ADVERSARIAL'):
    for mode in ('full', 'no_unlearn', 'no_recompute'):
        for n in (50, 100, 200):
            for a in (3, 5):
                for seed in range(1, 11):
                    g = generate_family_cfg(n, a, fam, seed, trap_depth=1)
                    f = {0} if fam == 'ADVERSARIAL' else choose_forgetting_request(g, 1, random.Random(seed + 10000))
                    disable(g, mode)
                    p, st = exact_ilp(g, f)
                    rows.append(dict(family=fam, mode=mode, n=n, actors=a, seed=seed, status=st,
                                     optimal=round(total_cost(g, p), 6),
                                     greedy=round(total_cost(g, greedy_repair(g, f)), 6),
                                     da1=round(total_cost(g, downstream_k_repair(g, f, 1.0, 1)), 6),
                                     unlearn_in_opt=sum(1 for v in affected_region(g, f) if p[v] == UNLEARN)))
        print(fam, mode, 'done', flush=True)
with open('results/results_ablation.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
