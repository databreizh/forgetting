"""Exact MCCF via CP-SAT (OR-Tools), matching feasible_actions() of cfe.py."""
from ortools.sat.python import cp_model
from collaborative_forgetting_experiments import (DELETE, INVALIDATE, RECOMPUTE, UNLEARN, KEEP, ACTIVE_ACTIONS,
                 affected_region, required_active_predecessors, total_cost, validate_plan)
SCALE = 10**6

def exact_ilp(g, forgotten, time_limit=600.0, workers=8):
    aff = affected_region(g, forgotten)
    m = cp_model.CpModel()
    x = {}
    plan = {v: KEEP for v in g.nodes if v not in aff}
    for v in aff:
        node = g.nodes[v]
        if v in forgotten: acts = [DELETE]
        elif node.kind == "input": acts = [KEEP]
        else:
            acts = [DELETE, INVALIDATE]
            if node.recomputable: acts.append(RECOMPUTE)
            if node.unlearnable: acts.append(UNLEARN)
        for a in acts: x[v, a] = m.NewBoolVar(f"x_{v}_{a}")
        m.AddExactlyOne(x[v, a] for a in acts)
    def active(u):
        if u not in aff: return 1
        return sum(x[u, a] for a in ACTIVE_ACTIONS if (u, a) in x)
    for v in aff:
        rep = [x[v, a] for a in (RECOMPUTE, UNLEARN) if (v, a) in x]
        if rep:
            req = required_active_predecessors(g, v)
            m.Add(sum(active(u) for u in g.pred[v]) >= req * sum(rep))
    m.Minimize(sum(int(round(SCALE * g.nodes[v].costs[a])) * var for (v, a), var in x.items()))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None, st
    for (v, a), var in x.items():
        if s.Value(var): plan[v] = a
    assert validate_plan(g, forgotten, plan)
    return plan, ("OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE")
