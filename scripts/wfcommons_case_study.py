"""Case study on real workflow traces (WfCommons / WfInstances, Pegasus).

Each task becomes an artifact whose recomputation cost is its measured runtime
(seconds). Raw input files (not produced by any task) become input nodes.
A forgetting request removes one raw input file.

Assumptions (not observable in the traces, stated in the paper):
  * only aggregation tasks (>= 3 inputs produced by tasks of the same kind)
    can be re-executed on a surviving subset of their inputs, provided a fraction
    THETA of them remain active; any other task needs all of its inputs;
  * losing an intermediate artifact costs RHO_INT x its runtime (cached work lost);
  * losing a final output (sink) costs the total runtime of its ancestors, i.e.
    the compute needed to regenerate that scientific result later.
Run from the repository root:
  python scripts/wfcommons_case_study.py --traces <WfInstances>/pegasus
"""
import sys, os, json, glob, csv, time, random, argparse, math, zlib
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import (CFG, Node, KEEP, DELETE, INVALIDATE, RECOMPUTE,
    affected_region, greedy_repair, downstream_k_repair, cascade_delete, total_cost,
    validate_plan, feasible_actions, ACTIVE_ACTIONS)
from mccf_ilp import exact_ilp

def load_trace(path, theta, rho_int):
    w = json.load(open(path))
    spec = w['workflow']['specification']; tasks = spec['tasks']
    rt = {x['id']: float(x.get('runtimeInSeconds', 0.0)) for x in w['workflow']['execution']['tasks']}
    produced = {o for t in tasks for o in t.get('outputFiles', [])}
    raw = sorted({i for t in tasks for i in t.get('inputFiles', []) if i not in produced})
    ids = {}
    g = CFG()
    for f in raw:
        ids[('f', f)] = len(ids)
        g.add_node(Node(id=ids[('f', f)], kind='input', owners={0}))
    for t in tasks:
        ids[('t', t['id'])] = len(ids)
        g.add_node(Node(id=ids[('t', t['id'])], kind='transform', owners={0}, recomputable=True))
    prod_by = {o: t['id'] for t in tasks for o in t.get('outputFiles', [])}
    for t in tasks:
        v = ids[('t', t['id'])]
        for i in t.get('inputFiles', []):
            u = ids[('f', i)] if i not in produced else ids[('t', prod_by[i])]
            g.add_edge(u, v)
        for p in t.get('parents', []):
            if ('t', p) in ids: g.add_edge(ids[('t', p)], v)
    order = g.topological_order()
    anc_rt = {}
    for v in order:          # total runtime of v and its task ancestors
        s = 0.0; seen = set(); st = [v]
        while st:
            x = st.pop()
            if x in seen: continue
            seen.add(x)
            st.extend(g.pred[x])
        anc_rt[v] = sum(g.nodes[x].costs.get('_rt', 0.0) for x in seen)  # filled below
    inv = {v: k for k, v in ids.items()}
    for v, node in g.nodes.items():
        r = rt.get(inv[v][1], 0.0) if node.kind == 'transform' else 0.0
        node.costs['_rt'] = r
    import re
    cat = {v: re.sub(r'_ID\d+$', '', inv[v][1]) for v in g.nodes if inv[v][0] == 't'}
    for v, node in g.nodes.items():
        node.costs[KEEP] = 0.0
        if node.kind == 'input':
            node.costs[DELETE] = node.costs[INVALIDATE] = 0.0
            continue
        r = node.costs['_rt']
        preds = list(g.pred[v])
        task_preds = [u for u in preds if u in cat]
        kinds = {cat[u] for u in task_preds}
        # Aggregation tasks (>= 3 inputs produced by tasks of one kind, e.g.
        # merge, mAdd) tolerate losing part of those inputs: they need all
        # other inputs plus a fraction THETA of the aggregated ones. Any other
        # task needs all of its inputs (an aligner needs its index and reads).
        if len(task_preds) >= 3 and len(kinds) == 1:
            need = (len(preds) - len(task_preds)) + math.ceil(theta * len(task_preds))
        else:
            need = len(preds)
        node.min_active_ratio = (need - 1e-9) / max(1, len(preds))
        node.costs[RECOMPUTE] = r
        if len(g.succ[v]) == 0:    # final output
            seen = set(); st = [v]
            while st:
                x = st.pop()
                if x in seen: continue
                seen.add(x); st.extend(g.pred[x])
            loss = sum(g.nodes[x].costs['_rt'] for x in seen)
        else:
            loss = rho_int * r
        node.costs[DELETE] = node.costs[INVALIDATE] = loss
    for node in g.nodes.values(): node.costs.pop('_rt', None)
    return g, ids, raw

def recompute_all(g, forgotten):
    aff = affected_region(g, forgotten); plan = {}
    for v in g.topological_order():
        if v in forgotten: plan[v] = DELETE
        elif v not in aff: plan[v] = KEEP
        else:
            feas = feasible_actions(g, v, plan, forgotten, aff)
            plan[v] = RECOMPUTE if RECOMPUTE in feas else min(
                (a for a in feas if a in (DELETE, INVALIDATE)), key=lambda a: g.nodes[v].costs[a])
    return plan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--theta', type=float, default=0.5)
    ap.add_argument('--rho', default='0.1,0.5,1.0')
    ap.add_argument('--requests', type=int, default=5)
    ap.add_argument('--max-k', type=int, default=3)
    ap.add_argument('--out', default='results/results_wfcommons.csv')
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.traces, '*', '*.json'))):
        for rho in [float(x) for x in a.rho.split(',')]:
            g, ids, raw = load_trace(path, a.theta, rho)
            cand = [ids[('f', f)] for f in raw if len(g.succ[ids[('f', f)]]) > 0]
            rng = random.Random(zlib.crc32(os.path.basename(path).encode()))
            cand = [c for c in cand if len(affected_region(g, {c})) >= 3]
            for s in rng.sample(cand, min(a.requests, len(cand))):
                f = {s}; aff = affected_region(g, f)
                rec = dict(workflow=path.split(os.sep)[-2], trace=os.path.basename(path),
                           tasks=sum(1 for n in g.nodes.values() if n.kind == 'transform'),
                           rho=rho, theta=a.theta, request=s, affected=len(aff))
                t0 = time.perf_counter(); p, st = exact_ilp(g, f, time_limit=600)
                rec['ilp_time'] = round(time.perf_counter() - t0, 4); rec['status'] = st
                rec['optimal'] = round(total_cost(g, p), 4)
                algos = [('cascade', lambda: cascade_delete(g, f)), ('greedy', lambda: greedy_repair(g, f)),
                         ('recompute_all', lambda: recompute_all(g, f))]
                algos += [(f'da{k}', (lambda k=k: downstream_k_repair(g, f, 1.0, k))) for k in range(1, a.max_k + 1)]
                for name, fn in algos:
                    t0 = time.perf_counter(); pl = fn(); dt = time.perf_counter() - t0
                    assert validate_plan(g, f, pl), name
                    rec[name] = round(total_cost(g, pl), 4); rec[name + '_time'] = round(dt, 4)
                    if name == 'cascade': continue
                pop = aff - f
                rec['retention_opt'] = round(sum(p[v] in ACTIVE_ACTIONS for v in pop) / max(1, len(pop)), 4)
                rows.append(rec)
            print(os.path.basename(path), rho, 'done', flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    main()
