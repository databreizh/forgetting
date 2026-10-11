"""Cost-sensitivity and RecomputeWheneverPossible baseline on the synthetic families.
Plans are computed on perturbed costs c'(v,a) = c(v,a) * exp(sigma * Z), Z ~ N(0,1),
and evaluated on the true costs. Run from the repository root:
  python scripts/run_sensitivity.py"""
import sys, csv, copy, math, random, zlib
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

def perturb(g, sigma, rng):
    h = copy.deepcopy(g)
    for nd in h.nodes.values():
        for a in list(nd.costs):
            if a != KEEP and nd.costs[a] > 0:
                nd.costs[a] *= math.exp(sigma * rng.gauss(0, 1))
    return h

ALGOS = {'exact': lambda g, f: exact_ilp(g, f)[0], 'greedy': greedy_repair,
         'rwp': recompute_whenever_possible, 'da1': lambda g, f: downstream_k_repair(g, f, 1.0, 1)}
rows = []
for fam in ('LOCAL', 'SHARED', 'ADVERSARIAL'):
    for n in (50, 100, 200):
        for a in (3, 5):
            for seed in range(1, 11):
                g = generate_family_cfg(n, a, fam, seed, trap_depth=1)
                f = {0} if fam == 'ADVERSARIAL' else choose_forgetting_request(g, 1, random.Random(seed + 10000))
                opt = total_cost(g, exact_ilp(g, f)[0])
                casc = total_cost(g, cascade_delete(g, f))
                for sigma in (0.0, 0.1, 0.25, 0.5, 1.0):
                    for rep in range(5 if sigma > 0 else 1):
                        rng = random.Random(zlib.crc32(repr((fam, n, a, seed, sigma, rep)).encode()))
                        h = perturb(g, sigma, rng) if sigma > 0 else g
                        rec = dict(family=fam, n=n, actors=a, seed=seed, sigma=sigma, rep=rep,
                                   optimal=round(opt, 6), cascade=round(casc, 6))
                        for name, fn in ALGOS.items():
                            p = fn(h, f); assert validate_plan(g, f, p)
                            rec[name] = round(total_cost(g, p), 6)   # true cost
                        rows.append(rec)
        print(fam, n, 'done', flush=True)
with open('results/results_sensitivity.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
