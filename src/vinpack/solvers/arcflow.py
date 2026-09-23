"""Arc-flow formulation for bin packing (Valério de Carvalho, 1999).

Nodes are partial truck loads 0..C. An item arc (v, v + s_t) places one item
of type t; loss arcs (v, C) close a truck. One unit of flow from 0 to C is one
truck. Item types are processed in decreasing size and a type-t arc may only
leave a node reachable by types >= t, which removes most symmetric packings.

Its LP relaxation equals the Gilmore-Gomory bound, but it is a compact model a
MIP solver can finish on its own. vinpack uses it as the exact fallback when
column generation + diving stop one truck above the proven lower bound.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from vinpack.io.instances import BinPackingInstance
from vinpack.solvers.backend import build, solve
from vinpack.solvers.bpp import BPPResult, lower_bound_l2


def build_graph(types: np.ndarray, demand: np.ndarray, C: int):
    order = np.argsort(-types, kind="stable")
    reach = np.zeros(C + 1, dtype=bool)
    reach[0] = True
    arcs: list[tuple[int, int, int]] = []  # (tail, head, type)
    for t in order:
        s, d = int(types[t]), int(demand[t])
        new_reach = reach.copy()
        # chains of up to d copies of type t starting from nodes reached by larger types
        frontier = np.flatnonzero(reach)
        seen: set[tuple[int, int]] = set()
        for v in frontier:
            u = int(v)
            for _ in range(d):
                if u + s > C:
                    break
                if (u, u + s) not in seen:
                    seen.add((u, u + s))
                    arcs.append((u, u + s, int(t)))
                u += s
                new_reach[u] = True
        reach = new_reach
    nodes = np.flatnonzero(reach)
    loss = [(int(v), C, -1) for v in nodes if v != C]
    return arcs + loss, nodes


def arcflow(inst: BinPackingInstance, backend: str = "highs", time_limit: float = 120.0,
            upper_bound: int | None = None) -> BPPResult:
    t0 = time.perf_counter()
    C = int(inst.capacity)
    types, inverse, demand = np.unique(inst.sizes, return_inverse=True, return_counts=True)
    arcs, nodes = build_graph(types, demand, C)
    tail = np.array([a[0] for a in arcs])
    head = np.array([a[1] for a in arcs])
    typ = np.array([a[2] for a in arcs])
    E = len(arcs)
    # variables: arc flows f_e (integer) + z (number of trucks)
    node_idx = {int(v): k for k, v in enumerate(nodes)}
    N = len(nodes)
    rows, cols, vals = [], [], []
    for e in range(E):
        rows += [node_idx[int(tail[e])], node_idx[int(head[e])]]
        cols += [e, e]
        vals += [-1.0, 1.0]
    # flow conservation: inflow - outflow = -z at 0, +z at C, 0 elsewhere
    rows += [node_idx[0], node_idx[C]]
    cols += [E, E]
    vals += [1.0, -1.0]
    r = N
    for t in range(len(types)):
        idx = np.flatnonzero(typ == t)
        rows += [r] * len(idx)
        cols += list(idx)
        vals += [1.0] * len(idx)
        r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, E + 1))
    row_lb = np.r_[np.zeros(N), demand.astype(float)]
    row_ub = np.r_[np.zeros(N), np.full(len(types), np.inf)]
    c = np.r_[np.zeros(E), 1.0]
    col_ub = np.full(E + 1, np.inf)
    if upper_bound is not None:
        col_ub[E] = upper_bound
    model = build(c, A, row_lb, row_ub, integer=True, col_ub=col_ub, sense="min")
    sol = solve(model, backend, time_limit)
    lb = max(lower_bound_l2(inst.sizes, C),
             int(np.ceil(sol.bound - 1e-6)) if np.isfinite(sol.bound) else 0)
    if not sol.feasible:
        return BPPResult([], 0, lb, f"arcflow-{backend}", time.perf_counter() - t0, sol.status,
                         extra={"arcs": E})
    bins = _decompose(inst, np.rint(sol.x[:E]).astype(int), tail, head, typ, C, inverse)
    status = "optimal" if len(bins) == lb else "feasible"
    return BPPResult(bins, len(bins), lb, f"arcflow-{backend}", time.perf_counter() - t0, status,
                     extra={"arcs": E, "nodes": N})


def _decompose(inst, flow, tail, head, typ, C, inverse) -> list[list[int]]:
    """Split an integer 0->C flow into paths (trucks) and assign concrete items."""
    out_arcs: dict[int, list[int]] = {}
    for e in np.flatnonzero(flow):
        out_arcs.setdefault(int(tail[e]), []).append(int(e))
    pools: dict[int, list[int]] = {}
    for i in range(inst.n):
        pools.setdefault(int(inverse[i]), []).append(i)
    bins = []
    while out_arcs.get(0):
        v, b = 0, []
        while v != C:
            e = next(e for e in out_arcs[v] if flow[e] > 0)
            flow[e] -= 1
            if flow[e] == 0:
                out_arcs[v].remove(e)
            if typ[e] >= 0 and pools.get(int(typ[e])):
                b.append(pools[int(typ[e])].pop())
            v = int(head[e])
        if b:
            bins.append(b)
    # items the flow over-covered are fine; items it missed cannot happen (demand rows)
    left = [i for lst in pools.values() for i in lst]
    assert not left, f"arc-flow decomposition left {len(left)} items unpacked"
    return bins
