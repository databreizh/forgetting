"""Multi-actor ablation: scale the actor (owner) penalty in SHARED CFGs and
measure how optimal plans react. Run from the repository root."""
import sys, csv, random, statistics as st
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import *
from mccf_ilp import exact_ilp

def reprice(g, w_owner):
    for v, nd in g.nodes.items():
        if nd.kind == 'input': continue
        f, o = len(g.succ[v]), len(nd.owners)
        pen = 0.9 * max(0, f - 1) + 0.8 * w_owner * max(0, o - 1)
        nd.costs[DELETE] = 1.2 + pen
        nd.costs[INVALIDATE] = 1.0 + 0.85 * pen

rows = []
for n in (50, 100, 200):
    for a in (3, 5):
        for seed in range(1, 11):
            base = None
            for w in (0.0, 1.0, 2.0, 4.0):
                g = generate_family_cfg(n, a, 'SHARED', seed)
                f = choose_forgetting_request(g, 1, random.Random(seed + 10000))
                reprice(g, w)
                p, _ = exact_ilp(g, f)
                aff = affected_region(g, f) - f
                multi = [v for v in aff if len(g.nodes[v].owners) > 1]
                single = [v for v in aff if len(g.nodes[v].owners) == 1]
                ret = lambda S: sum(p[v] in ACTIVE_ACTIONS for v in S) / len(S) if S else float('nan')
                if base is None: base = p
                rows.append(dict(n=n, actors=a, seed=seed, w_owner=w, affected=len(aff),
                    optimal=round(total_cost(g, p), 6),
                    greedy=round(total_cost(g, greedy_repair(g, f)), 6),
                    da1=round(total_cost(g, downstream_k_repair(g, f, 1.0, 1)), 6),
                    n_multi=len(multi), ret_multi=ret(multi), ret_single=ret(single),
                    changed=sum(p[v] != base[v] for v in aff) / max(1, len(aff))))
with open('results/results_actor_ablation.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
