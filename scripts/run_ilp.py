import sys, csv, time
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import *
from mccf_ilp import exact_ilp
rows = []
for d in (1, 2, 3):
    for n in (50, 100, 200):
        for a in (3, 5):
            for seed in range(1, 11):
                g = generate_family_cfg(n, a, 'ADVERSARIAL', seed, trap_depth=d); f = {0}
                t = time.perf_counter(); p, st = exact_ilp(g, f, time_limit=300); ti = time.perf_counter() - t
                opt = total_cost(g, p)
                r = dict(trap_depth=d, n=n, actors=a, seed=seed, affected=len(affected_region(g, f)),
                         status=st, ilp_time=round(ti, 3), optimal=round(opt, 6))
                r['greedy'] = round(total_cost(g, greedy_repair(g, f)), 6)
                for k in (1, 2, 3):
                    r[f'da{k}'] = round(total_cost(g, downstream_k_repair(g, f, 1.0, k)), 6)
                rows.append(r)
                print(d, n, a, seed, st, r['affected'], round(ti, 2), flush=True)
with open('results/results_ilp.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
