#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

KEEP = "keep"
DELETE = "delete"
INVALIDATE = "invalidate"
RECOMPUTE = "recompute"
UNLEARN = "unlearn"
ACTIVE_ACTIONS = {KEEP, RECOMPUTE, UNLEARN}


@dataclass
class Node:
    id: int
    kind: str
    owners: Set[int]
    recomputable: bool = False
    unlearnable: bool = False
    min_active_ratio: float = 1.0
    costs: Dict[str, float] = field(default_factory=dict)


class CFG:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.succ: Dict[int, Set[int]] = {}
        self.pred: Dict[int, Set[int]] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        self.succ.setdefault(node.id, set())
        self.pred.setdefault(node.id, set())

    def add_edge(self, u: int, v: int):
        if u == v:
            return
        self.succ.setdefault(u, set()).add(v)
        self.pred.setdefault(v, set()).add(u)
        self.succ.setdefault(v, set())
        self.pred.setdefault(u, set())

    def remove_all_in_edges(self, v: int):
        for u in list(self.pred[v]):
            self.succ[u].discard(v)
        self.pred[v].clear()

    def descendants(self, sources: Set[int]) -> Set[int]:
        seen = set(sources)
        stack = list(sources)
        while stack:
            u = stack.pop()
            for v in self.succ.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen - set(sources)

    def topological_order(self) -> List[int]:
        indeg = {v: len(self.pred[v]) for v in self.nodes}
        q = sorted(v for v, d in indeg.items() if d == 0)
        out = []
        i = 0
        while i < len(q):
            u = q[i]
            i += 1
            out.append(u)
            for v in self.succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(out) != len(self.nodes):
            raise ValueError("Graph is not acyclic")
        return out


def sharing_params(sharing: str) -> Tuple[float, float]:
    if sharing == "low":
        return 0.15, 0.05
    if sharing == "medium":
        return 0.45, 0.25
    if sharing == "high":
        return 1.0, 0.55
    raise ValueError(sharing)


def generate_cfg(
    n: int,
    actors: int,
    sharing: str,
    seed: int,
    layers: int = 8,
    recompute_prob: float = 0.75,
    unlearn_prob: float = 0.60,
) -> CFG:
    rng = random.Random(seed)
    g = CFG()
    extra_edge_factor, multi_owner_prob = sharing_params(sharing)

    layers = max(3, min(layers, n))
    input_count = max(2, min(int(0.15 * n), n // 3))
    layer_of: Dict[int, int] = {}

    for i in range(n):
        if i < input_count:
            layer = 0
            kind = "input"
        else:
            layer = 1 + rng.randrange(layers - 1)
            x = rng.random()
            if layer == layers - 1 and x < 0.55:
                kind = "output"
            elif x < 0.25:
                kind = "composite"
            elif x < 0.43:
                kind = "model"
            else:
                kind = "transform"

        layer_of[i] = layer
        owners = {rng.randrange(actors)}
        if actors > 1 and rng.random() < multi_owner_prob:
            owners.add(rng.randrange(actors))
        if actors > 2 and sharing == "high" and rng.random() < 0.15:
            owners.add(rng.randrange(actors))

        recomputable = kind != "input" and rng.random() < recompute_prob
        unlearnable = kind == "model" and rng.random() < unlearn_prob

        if kind == "composite":
            min_ratio = rng.choice([0.50, 0.67, 0.75, 1.0])
        elif kind == "model":
            min_ratio = rng.choice([0.67, 0.75, 1.0])
        else:
            min_ratio = rng.choice([0.75, 1.0])

        g.add_node(Node(
            id=i,
            kind=kind,
            owners=owners,
            recomputable=recomputable,
            unlearnable=unlearnable,
            min_active_ratio=min_ratio,
        ))

    for v in range(input_count, n):
        lv = layer_of[v]
        candidates = [u for u in range(v) if layer_of[u] < lv]
        if not candidates:
            candidates = list(range(input_count))
        base = {"low": 1, "medium": 2, "high": 3}[sharing]
        k = min(len(candidates), rng.randint(1, max(1, base + 1)))
        for u in rng.sample(candidates, k):
            g.add_edge(u, v)

    extra_edges = int(n * extra_edge_factor)
    attempts = 0
    added = 0
    while added < extra_edges and attempts < extra_edges * 20 + 100:
        attempts += 1
        u = rng.randrange(n)
        v = rng.randrange(n)
        if layer_of[u] < layer_of[v] and v not in g.succ[u]:
            g.add_edge(u, v)
            added += 1

    assign_costs(g, rng)
    return g


def assign_costs(g: CFG, rng: random.Random):
    for v, node in g.nodes.items():
        consumer_count = max(1, len(g.succ[v]))
        owner_count = max(1, len(node.owners))
        sharing_penalty = 0.35 * (consumer_count - 1) + 0.50 * (owner_count - 1)
        node.costs[KEEP] = 0.0
        node.costs[DELETE] = 1.0 + sharing_penalty
        node.costs[INVALIDATE] = 0.85 + 0.8 * sharing_penalty
        if node.recomputable:
            node.costs[RECOMPUTE] = rng.uniform(0.6, 3.0)
        if node.unlearnable:
            node.costs[UNLEARN] = rng.uniform(0.5, 2.5)


def configure_family(g: CFG, family: str, rng: random.Random) -> None:
    family = family.upper()
    for v, node in g.nodes.items():
        fanout = len(g.succ[v])
        owners = len(node.owners)
        shared = fanout >= 2 or owners >= 2

        if family == "LOCAL":
            node.costs[DELETE] = 1.0 + 0.15 * max(0, fanout - 1)
            node.costs[INVALIDATE] = 0.9 + 0.12 * max(0, fanout - 1)
            if node.recomputable:
                node.costs[RECOMPUTE] = rng.uniform(0.75, 1.25)
            if node.unlearnable:
                node.costs[UNLEARN] = rng.uniform(0.7, 1.2)

        elif family == "SHARED":
            penalty = 0.9 * max(0, fanout - 1) + 0.8 * max(0, owners - 1)
            node.costs[DELETE] = 1.2 + penalty
            node.costs[INVALIDATE] = 1.0 + 0.85 * penalty
            if node.recomputable:
                node.costs[RECOMPUTE] = rng.uniform(0.65, 1.55)
            if node.unlearnable:
                node.costs[UNLEARN] = rng.uniform(0.6, 1.4)

        elif family == "ADVERSARIAL":
            node.costs[DELETE] = 0.75
            node.costs[INVALIDATE] = 0.60
            if node.recomputable:
                node.costs[RECOMPUTE] = rng.uniform(0.95, 1.25)
            if node.unlearnable:
                node.costs[UNLEARN] = rng.uniform(0.9, 1.2)

            if shared and node.kind != "input":
                node.costs[DELETE] = 3.0 + 0.8 * fanout
                node.costs[INVALIDATE] = 2.6 + 0.7 * fanout
                if node.recomputable:
                    node.costs[RECOMPUTE] = rng.uniform(0.55, 0.95)
                if node.unlearnable:
                    node.costs[UNLEARN] = rng.uniform(0.5, 0.9)
        else:
            raise ValueError(f"Unknown family: {family}")


def _embed_adversarial_traps(g: CFG, trap_depth: int) -> None:
    """Embed explicit depth-k traps rooted at the forgotten source 0.

    For each gadget, the root p has one forgotten parent (0) and one clean
    backup source (1), hence it can be recomputed after forgetting 0.
    Greedy prefers cheap invalidation at p.  A chain of `trap_depth`
    descendants follows. Intermediate descendants are cheap regardless of
    whether they stay active, while the terminal descendant is expensive to
    lose. Therefore a look-ahead shorter than trap_depth cannot fully see the
    terminal penalty.
    """
    if trap_depth < 1:
        raise ValueError("trap_depth must be >= 1")

    topo = g.topological_order()
    non_inputs = [v for v in topo if g.nodes[v].kind != "input"]
    per_gadget = trap_depth + 1  # root + chain nodes
    trap_count = max(1, min(5, len(non_inputs) // max(1, 2 * per_gadget)))
    needed = trap_count * per_gadget
    if len(non_inputs) < needed:
        trap_count = max(1, len(non_inputs) // per_gadget)
    if trap_count == 0:
        return

    chosen = non_inputs[: trap_count * per_gadget]
    clean = 1 if 1 in g.nodes else None

    for t in range(trap_count):
        block = chosen[t * per_gadget:(t + 1) * per_gadget]
        if len(block) < per_gadget:
            break
        root = block[0]
        chain = block[1:]

        # Root: old materialization depends on forgotten 0, but can be rebuilt
        # from clean source 1. This mirrors the corrected NP-hardness gadget.
        g.remove_all_in_edges(root)
        g.add_edge(0, root)
        if clean is not None:
            g.add_edge(clean, root)
        g.nodes[root].recomputable = True
        g.nodes[root].unlearnable = False
        g.nodes[root].min_active_ratio = 0.5 if clean is not None else 1.0
        g.nodes[root].costs[DELETE] = 0.75
        g.nodes[root].costs[INVALIDATE] = 0.60
        g.nodes[root].costs[RECOMPUTE] = 1.10

        parent = root
        for depth_idx, node_id in enumerate(chain, start=1):
            g.remove_all_in_edges(node_id)
            g.add_edge(parent, node_id)
            node = g.nodes[node_id]
            node.recomputable = True
            node.unlearnable = False
            node.min_active_ratio = 1.0

            if depth_idx < trap_depth:
                # Neutral intermediate stage: no visible penalty yet.
                node.costs[DELETE] = 0.20
                node.costs[INVALIDATE] = 0.20
                node.costs[RECOMPUTE] = 0.20
            else:
                # Terminal penalty only becomes visible at requested depth.
                node.costs[DELETE] = 5.00
                node.costs[INVALIDATE] = 4.50
                node.costs[RECOMPUTE] = 0.20
            parent = node_id


def generate_family_cfg(
    n: int,
    actors: int,
    family: str,
    seed: int,
    trap_depth: int = 1,
) -> CFG:
    family = family.upper()
    if family == "LOCAL":
        g = generate_cfg(n, actors, "low", seed, recompute_prob=0.70, unlearn_prob=0.50)
    elif family == "SHARED":
        g = generate_cfg(n, actors, "high", seed, recompute_prob=0.85, unlearn_prob=0.70)
    elif family == "ADVERSARIAL":
        g = generate_cfg(n, actors, "high", seed, recompute_prob=0.92, unlearn_prob=0.75)
        for node in g.nodes.values():
            if node.kind != "input":
                node.min_active_ratio = min(node.min_active_ratio, 0.67)
    else:
        raise ValueError(family)

    configure_family(g, family, random.Random(seed + 424242))
    if family == "ADVERSARIAL":
        _embed_adversarial_traps(g, trap_depth)
    return g


def choose_forgetting_request(g: CFG, size: int, rng: random.Random) -> Set[int]:
    candidates = [v for v, node in g.nodes.items() if node.kind == "input" and len(g.succ[v]) > 0]
    size = min(size, len(candidates))
    return set(rng.sample(candidates, size))


def affected_region(g: CFG, forgotten: Set[int]) -> Set[int]:
    return set(forgotten) | g.descendants(forgotten)


def _remove_edge(g: CFG, u: int, v: int) -> None:
    g.succ.get(u, set()).discard(v)
    g.pred.get(v, set()).discard(u)


def force_controlled_affected_region(
    g: CFG,
    fraction: float,
    source: int = 0,
) -> Set[int]:
    """Force a forgetting request whose affected region is about fraction*n.

    This helper is intended for scalability experiments, where runtime should be
    studied as a function of the *affected* subgraph rather than only total CFG
    size.  It constructs a controlled reachable region from `source` while
    preserving acyclicity:

      1. choose target nodes in the existing topological order;
      2. remove edges from the target region to nodes outside it, so reachability
         cannot leak beyond the requested region;
      3. add a topologically ordered backbone from `source` through the chosen
         nodes, ensuring they are all affected.

    Incoming edges from outside the region are preserved.  They act as clean,
    surviving predecessors and make the resulting workload closer to the repair
    setting studied in the paper.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("affected fraction must be in (0,1]")
    if source not in g.nodes:
        raise ValueError(f"source {source} is not in the CFG")
    if g.nodes[source].kind != "input":
        raise ValueError("controlled affected-region source must be an input node")

    n = len(g.nodes)
    target_size = max(2, min(n, int(round(fraction * n))))
    topo = g.topological_order()
    pos = {v: i for i, v in enumerate(topo)}

    # Prefer non-input artifacts so the affected region models downstream repair.
    candidates = [
        v for v in topo
        if v != source and pos[v] > pos[source] and g.nodes[v].kind != "input"
    ]
    if len(candidates) < target_size - 1:
        candidates = [v for v in topo if v != source and pos[v] > pos[source]]

    chosen = candidates[: max(0, target_size - 1)]
    region = {source, *chosen}

    # Prevent descendants of the source from escaping the controlled region.
    for u in list(region):
        for v in list(g.succ.get(u, set())):
            if v not in region:
                _remove_edge(g, u, v)

    # Also remove any pre-existing direct source edges outside the region.
    for v in list(g.succ.get(source, set())):
        if v not in region:
            _remove_edge(g, source, v)

    # Ensure every selected node is reachable from the forgotten source while
    # preserving the existing topological order and therefore acyclicity.
    previous = source
    for v in chosen:
        g.add_edge(previous, v)
        previous = v

    forgotten = {source}
    actual = affected_region(g, forgotten)

    # The construction should be exact.  Fail loudly rather than silently
    # reporting a misleading scalability point.
    if actual != region:
        extra = len(actual - region)
        missing = len(region - actual)
        raise RuntimeError(
            f"controlled affected-region construction failed: "
            f"target={len(region)}, actual={len(actual)}, extra={extra}, missing={missing}"
        )
    return forgotten


def choose_forgetting_request_by_fraction(
    g: CFG,
    fraction: float,
    controlled: bool,
    rng: random.Random,
) -> Set[int]:
    """Choose a forgetting request for affected-region scalability studies.

    With controlled=True, the CFG is minimally reshaped so that source 0 has an
    affected region of the requested fraction.  With controlled=False, the input
    whose natural affected region is closest to the target is selected instead.
    """
    if controlled:
        return force_controlled_affected_region(g, fraction, source=0)

    target = max(1, int(round(fraction * len(g.nodes))))
    candidates = [
        v for v, node in g.nodes.items()
        if node.kind == "input" and len(g.succ[v]) > 0
    ]
    if not candidates:
        return choose_forgetting_request(g, 1, rng)

    scored = []
    for v in candidates:
        size = len(affected_region(g, {v}))
        scored.append((abs(size - target), -size, v))
    scored.sort()
    return {scored[0][2]}


def active(plan: Dict[int, str], v: int) -> bool:
    return plan.get(v) in ACTIVE_ACTIONS


def required_active_predecessors(g: CFG, v: int) -> int:
    p = len(g.pred[v])
    if p == 0:
        return 0
    return max(1, math.ceil(g.nodes[v].min_active_ratio * p))


def feasible_actions(
    g: CFG,
    v: int,
    plan: Dict[int, str],
    forgotten: Set[int],
    affected: Set[int],
) -> List[str]:
    node = g.nodes[v]
    if v in forgotten:
        return [DELETE]
    if node.kind == "input":
        return [KEEP]

    feasible = [DELETE, INVALIDATE]
    if v not in affected:
        feasible.append(KEEP)

    active_preds = sum(1 for u in g.pred[v] if active(plan, u))
    required = required_active_predecessors(g, v)
    if node.recomputable and active_preds >= required:
        feasible.append(RECOMPUTE)
    if node.unlearnable and active_preds >= required:
        feasible.append(UNLEARN)
    return feasible


def provisional_plan_for_node(
    g: CFG,
    node_id: int,
    plan: Dict[int, str],
) -> Dict[int, str]:
    """Return a copy of plan with undecided predecessors provisionally active.

    This implements the paper's optimistic convention in Sec. 6.4.
    """
    tmp = dict(plan)
    for p in g.pred[node_id]:
        if p not in tmp:
            tmp[p] = KEEP
    return tmp


def provisional_feasible_actions(
    g: CFG,
    v: int,
    plan: Dict[int, str],
    forgotten: Set[int],
    affected: Set[int],
) -> List[str]:
    return feasible_actions(g, v, provisional_plan_for_node(g, v, plan), forgotten, affected)


def total_cost(g: CFG, plan: Dict[int, str]) -> float:
    return sum(g.nodes[v].costs[a] for v, a in plan.items())


def cascade_delete(g: CFG, forgotten: Set[int]) -> Dict[int, str]:
    affected = affected_region(g, forgotten)
    return {v: DELETE if v in affected else KEEP for v in g.nodes}


def greedy_repair(g: CFG, forgotten: Set[int]) -> Dict[int, str]:
    affected = affected_region(g, forgotten)
    plan: Dict[int, str] = {}
    for v in g.topological_order():
        if v in forgotten:
            plan[v] = DELETE
        elif v not in affected:
            plan[v] = KEEP
        else:
            feas = feasible_actions(g, v, plan, forgotten, affected)
            if not feas:
                raise RuntimeError(f"No feasible action for node {v}")
            plan[v] = min(feas, key=lambda a: g.nodes[v].costs[a])
    return plan


def _future_cost_recursive(
    g: CFG,
    node_id: int,
    candidate: str,
    plan: Dict[int, str],
    forgotten: Set[int],
    affected: Set[int],
    depth: int,
) -> float:
    """Optimistic k-step downstream estimate after assigning candidate to node.

    depth=1 reproduces the paper's one-step estimate: immediate successors only.
    For depth>1, the estimate recursively chooses, at each successor, the
    provisional action minimizing its immediate cost plus the estimated cost of
    descendants up to the remaining horizon.

    Important: lambda is *not* applied recursively. It weights the complete
    downstream estimate once in downstream_k_repair(), so lambda has the clean
    interpretation used in the SIGMOD paper:
        score_k(v,a) = c(v,a) + lambda * D_hat_k(v,a).
    """
    if depth <= 0:
        return 0.0

    tmp = dict(plan)
    tmp[node_id] = candidate
    total = 0.0

    for u in g.succ[node_id]:
        if u not in affected:
            continue

        feas = provisional_feasible_actions(g, u, tmp, forgotten, affected)
        if not feas:
            return float("inf")

        best = float("inf")
        for b in feas:
            estimated = g.nodes[u].costs[b]
            if depth > 1:
                estimated += _future_cost_recursive(
                    g, u, b, tmp, forgotten, affected, depth - 1
                )
            best = min(best, estimated)

        total += best

    return total


def downstream_k_repair(
    g: CFG,
    forgotten: Set[int],
    lam: float = 1.0,
    lookahead_depth: int = 1,
) -> Dict[int, str]:
    if lookahead_depth < 1:
        raise ValueError("lookahead_depth must be >= 1")

    affected = affected_region(g, forgotten)
    plan: Dict[int, str] = {}

    for v in g.topological_order():
        if v in forgotten:
            plan[v] = DELETE
            continue
        if v not in affected:
            plan[v] = KEEP
            continue

        feas = feasible_actions(g, v, plan, forgotten, affected)
        if not feas:
            raise RuntimeError(f"No feasible action for node {v}")

        def score(a: str) -> float:
            future = _future_cost_recursive(
                g, v, a, plan, forgotten, affected, lookahead_depth
            )
            return g.nodes[v].costs[a] + lam * future

        plan[v] = min(feas, key=score)
    return plan


def downstream_aware_repair(g: CFG, forgotten: Set[int], lam: float = 1.0) -> Dict[int, str]:
    return downstream_k_repair(g, forgotten, lam=lam, lookahead_depth=1)


def exact_branch_and_bound(
    g: CFG,
    forgotten: Set[int],
    max_affected: int = 22,
) -> Optional[Dict[int, str]]:
    affected = affected_region(g, forgotten)
    if len(affected) > max_affected:
        return None

    order = [v for v in g.topological_order() if v in affected]
    plan = {v: KEEP for v in g.nodes if v not in affected}
    best_plan = None
    best_cost = float("inf")

    min_raw = {}
    for v in order:
        node = g.nodes[v]
        if v in forgotten:
            min_raw[v] = node.costs[DELETE]
        else:
            costs = [node.costs[DELETE], node.costs[INVALIDATE]]
            if node.recomputable:
                costs.append(node.costs[RECOMPUTE])
            if node.unlearnable:
                costs.append(node.costs[UNLEARN])
            min_raw[v] = min(costs)

    suffix_lb = [0.0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix_lb[i] = suffix_lb[i + 1] + min_raw[order[i]]

    def rec(i: int, cost_so_far: float):
        nonlocal best_plan, best_cost
        if cost_so_far + suffix_lb[i] >= best_cost:
            return
        if i == len(order):
            best_cost = cost_so_far
            best_plan = dict(plan)
            return

        v = order[i]
        feas = feasible_actions(g, v, plan, forgotten, affected)
        feas.sort(key=lambda a: g.nodes[v].costs[a])
        for a in feas:
            c = g.nodes[v].costs[a]
            if cost_so_far + c >= best_cost:
                continue
            plan[v] = a
            rec(i + 1, cost_so_far + c)
            del plan[v]

    rec(0, 0.0)
    return best_plan


def validate_plan(g: CFG, forgotten: Set[int], plan: Dict[int, str]) -> bool:
    affected = affected_region(g, forgotten)
    partial: Dict[int, str] = {}
    for v in g.topological_order():
        if v not in plan:
            return False
        if plan[v] not in feasible_actions(g, v, partial, forgotten, affected):
            return False
        partial[v] = plan[v]
    return True


def metrics(
    g: CFG,
    forgotten: Set[int],
    plan: Dict[int, str],
    runtime_s: float,
    optimal_cost: Optional[float],
):
    affected = affected_region(g, forgotten)
    cost = total_cost(g, plan)

    # Artifact loss keeps the historical definition over the full affected
    # region. Retention follows the paper definition and excludes the
    # explicitly forgotten request set S from both numerator and denominator:
    #
    #   Retention(pi) =
    #     |{v in V_R \ S : active_pi(v)}| / |V_R \ S|.
    #
    # This avoids mechanically penalizing every plan for nodes that the
    # forgetting request requires to be deleted.
    loss = sum(plan[v] in {DELETE, INVALIDATE} for v in affected)

    retention_population = affected - forgotten
    retained = sum(plan[v] in ACTIVE_ACTIONS for v in retention_population)
    retention = (
        retained / len(retention_population)
        if retention_population
        else 1.0
    )

    if optimal_cost is None:
        gap = ""
    elif optimal_cost == 0:
        gap = 0.0
    else:
        gap = (cost - optimal_cost) / optimal_cost

    counts = {a: sum(plan[v] == a for v in affected)
              for a in [KEEP, DELETE, INVALIDATE, RECOMPUTE, UNLEARN]}
    return {
        "cost": cost,
        "loss": loss,
        "retention": retention,
        "runtime_s": runtime_s,
        "optimal_cost": "" if optimal_cost is None else optimal_cost,
        "gap": gap,
        "keep": counts[KEEP],
        "delete": counts[DELETE],
        "invalidate": counts[INVALIDATE],
        "recompute": counts[RECOMPUTE],
        "unlearn": counts[UNLEARN],
        "affected_size": len(affected),
    }


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def run_experiment(
    n_values: List[int],
    actors_values: List[int],
    family_values: List[str],
    seeds: List[int],
    request_size: int,
    lambdas: List[float],
    lookahead_depths: List[int],
    trap_depths: List[int],
    exact_limit: int,
    output_csv: str,
    affected_fractions: Optional[List[float]] = None,
    controlled_affected: bool = False,
):
    rows = []

    for n in n_values:
        for actors in actors_values:
            for family in family_values:
                family_upper = family.upper()
                td_values = trap_depths if family_upper == "ADVERSARIAL" else [0]
                fraction_values = affected_fractions if affected_fractions else [None]
                for trap_depth in td_values:
                    for affected_fraction in fraction_values:
                        for seed in seeds:
                            g = generate_family_cfg(
                                n, actors, family_upper, seed,
                                trap_depth=max(1, trap_depth) if family_upper == "ADVERSARIAL" else 1,
                            )
                            rng_request = random.Random(seed + 10000)
                            if affected_fraction is not None:
                                forgotten = choose_forgetting_request_by_fraction(
                                    g, affected_fraction, controlled_affected, rng_request
                                )
                            elif family_upper == "ADVERSARIAL":
                                forgotten = {0}
                            else:
                                forgotten = choose_forgetting_request(
                                    g, request_size, rng_request
                                )
                            affected = affected_region(g, forgotten)
                            actual_fraction = len(affected) / max(1, len(g.nodes))

                            exact_plan = None
                            exact_time = float("nan")
                            optimal_cost = None
                            if len(affected) <= exact_limit:
                                exact_plan, exact_time = timed(
                                    exact_branch_and_bound, g, forgotten, exact_limit
                                )
                                if exact_plan is not None:
                                    assert validate_plan(g, forgotten, exact_plan)
                                    optimal_cost = total_cost(g, exact_plan)

                            base_methods = [
                                ("cascade", cascade_delete, {}),
                                ("greedy", greedy_repair, {}),
                            ]

                            for name, fn, kwargs in base_methods:
                                plan, runtime = timed(fn, g, forgotten, **kwargs)
                                if not validate_plan(g, forgotten, plan):
                                    raise RuntimeError(
                                        f"Invalid plan: {name}, n={n}, seed={seed}, family={family}"
                                    )
                                m = metrics(g, forgotten, plan, runtime, optimal_cost)
                                rows.append(_row(seed, n, actors, family_upper, trap_depth,
                                                 len(forgotten), name, "", "", m, affected_fraction, actual_fraction))

                            for depth in lookahead_depths:
                                for lam in lambdas:
                                    name = f"downstream_k{depth}"
                                    plan, runtime = timed(
                                        downstream_k_repair, g, forgotten,
                                        lam=lam, lookahead_depth=depth
                                    )
                                    if not validate_plan(g, forgotten, plan):
                                        raise RuntimeError(
                                            f"Invalid plan: {name}, n={n}, seed={seed}, family={family}, "
                                            f"trap_depth={trap_depth}, lambda={lam}"
                                        )
                                    m = metrics(g, forgotten, plan, runtime, optimal_cost)
                                    rows.append(_row(seed, n, actors, family_upper, trap_depth,
                                                     len(forgotten), name, lam, depth, m, affected_fraction, actual_fraction))

                            if exact_plan is not None:
                                m = metrics(g, forgotten, exact_plan, exact_time, optimal_cost)
                                rows.append(_row(seed, n, actors, family_upper, trap_depth,
                                                 len(forgotten), "optimal", "", "", m, affected_fraction, actual_fraction))

                            print(
                                f"done n={n:6d} actors={actors:2d} family={family_upper:11s} "
                                f"trap={trap_depth:2d} seed={seed:3d} affected={len(affected):7d} "
                                f"fraction={actual_fraction:.3f}"
                            )

    fields = [
        "seed", "n", "actors", "family", "trap_depth", "request_size",
        "affected_fraction_target", "affected_fraction_actual",
        "affected_size", "algorithm", "lambda", "lookahead_depth",
        "cost", "loss", "retention", "runtime_s", "optimal_cost", "gap",
        "keep", "delete", "invalidate", "recompute", "unlearn",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to {output_csv}")
    print(f"{len(rows)} rows")


def _row(seed, n, actors, family, trap_depth, request_size,
         algorithm, lam, lookahead_depth, m,
         affected_fraction_target=None, affected_fraction_actual=None):
    return {
        "seed": seed,
        "n": n,
        "actors": actors,
        "family": family,
        "trap_depth": trap_depth,
        "request_size": request_size,
        "affected_fraction_target": "" if affected_fraction_target is None else affected_fraction_target,
        "affected_fraction_actual": "" if affected_fraction_actual is None else round(affected_fraction_actual, 6),
        "affected_size": m["affected_size"],
        "algorithm": algorithm,
        "lambda": lam,
        "lookahead_depth": lookahead_depth,
        "cost": round(m["cost"], 6),
        "loss": m["loss"],
        "retention": round(m["retention"], 6),
        "runtime_s": round(m["runtime_s"], 9),
        "optimal_cost": round(m["optimal_cost"], 6) if m["optimal_cost"] != "" else "",
        "gap": round(m["gap"], 6) if m["gap"] != "" else "",
        "keep": m["keep"],
        "delete": m["delete"],
        "invalidate": m["invalidate"],
        "recompute": m["recompute"],
        "unlearn": m["unlearn"],
    }


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Collaborative Forgetting final experiments with depth-k traps, "
                    "k-step look-ahead, lambda sensitivity, and controlled affected-region scalability."
    )
    p.add_argument("--n", default="50,100,200")
    p.add_argument("--actors", default="3,5")
    p.add_argument("--families", default="LOCAL,SHARED,ADVERSARIAL")
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--request-size", type=int, default=1)
    p.add_argument("--lambdas", default="1.0",
                   help="Comma-separated lambda values, e.g. 0,0.25,0.5,1,2,4")
    p.add_argument("--lookahead-depths", default="1",
                   help="Comma-separated k values, e.g. 1,2,3")
    p.add_argument("--trap-depths", default="1",
                   help="Comma-separated adversarial trap depths, e.g. 1,2,3")
    p.add_argument("--exact-limit", type=int, default=22)
    p.add_argument(
        "--affected-fractions", default="",
        help="Comma-separated target fractions of the CFG to place in the affected region, "
             "e.g. 0.1,0.25,0.5. Empty keeps the original request-selection behavior."
    )
    p.add_argument(
        "--controlled-affected", action="store_true",
        help="Force the requested affected-region fractions by shaping reachability from source 0. "
             "Recommended for scalability experiments."
    )
    p.add_argument("--out", default="results_collaborative_forgetting.csv")
    args = p.parse_args()

    run_experiment(
        parse_ints(args.n),
        parse_ints(args.actors),
        parse_strs(args.families),
        parse_ints(args.seeds),
        args.request_size,
        parse_floats(args.lambdas),
        parse_ints(args.lookahead_depths),
        parse_ints(args.trap_depths),
        args.exact_limit,
        args.out,
        parse_floats(args.affected_fractions) if args.affected_fractions.strip() else None,
        args.controlled_affected,
    )


if __name__ == "__main__":
    main()
