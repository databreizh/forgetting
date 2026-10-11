"""Repair-cost ratio sweep: recompute/unlearn costs scaled by kappa (synthetic families).
Run from the repository root:  python scripts/run_repair_ratio.py"""
import sys, csv, random
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import *
from mccf_ilp import exact_ilp

def recompute_whenever_possible(g, f):
    aff = affected_region(g, f); plan = {}
    for v in g.topological_order():
        if v in f: plan[v] = DELETE
        elif v not in aff: plan[v] = KEEP
        else:
            feas = feasible_actions(g, v, plan, f, aff)
            rep = [a for a in feas if a in (RECOMPUTE, UNLEARN)]
            pool = rep if rep else [a for a in feas if a in (DELETE, INVALIDATE)]
            plan[v] = min(pool, key=lambda a: g.nodes[v].costs[a])
    return plan
def main():
    rows = []
    for fam in ('LOCAL', 'SHARED', 'ADVERSARIAL'):
        for kappa in (0.5, 1.0, 2.0, 4.0):
            for n in (50, 100, 200):
                for a in (3, 5):
                    for seed in range(1, 11):
                        g = generate_family_cfg(n, a, fam, seed, trap_depth=1)
                        f = {0} if fam == 'ADVERSARIAL' else choose_forgetting_request(g, 1, random.Random(seed + 10000))
                        for nd in g.nodes.values():
                            for act in (RECOMPUTE, UNLEARN):
                                if act in nd.costs: nd.costs[act] *= kappa
                        rec = dict(family=fam, kappa=kappa, n=n, actors=a, seed=seed,
                                   optimal=total_cost(g, exact_ilp(g, f)[0]),
                                   rwp=total_cost(g, recompute_whenever_possible(g, f)),
                                   greedy=total_cost(g, greedy_repair(g, f)),
                                   da1=total_cost(g, downstream_k_repair(g, f, 1.0, 1)),
                                   cascade=total_cost(g, cascade_delete(g, f)))
                        rows.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()})
            print(fam, kappa, flush=True)
    with open('results/results_repair_ratio.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
