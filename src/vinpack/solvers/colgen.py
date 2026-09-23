"""Gilmore-Gomory column generation for bin packing, priced by the L0 knapsack kernel.

Items with equal size are aggregated into types t with demand d_t. A column is
a feasible truck load ("pattern") a_t = copies of type t on one truck.

    Master LP:  min sum_p lambda_p   s.t.  sum_p a_tp lambda_p >= d_t,  lambda >= 0
    Pricing:    max sum_t pi_t a_t   s.t.  sum_t s_t a_t <= C,  0 <= a_t <= min(d_t, C // s_t)

Pricing is a bounded knapsack; it is solved exactly with the 0-1 DP (``kp_dp``)
over binary-split copies of each type. A pattern prices out while its value
exceeds 1. The LP optimum z* gives the lower bound ceil(z*), which is tight for
almost all practical instances (the IRUP property), so rounding the master
often *proves* optimality - something FFD can never do.

Integer phase: (1) round the LP down and FFD the residual, (2) solve the
restricted master as an integer program over the generated columns, and if
neither reaches ceil(z*), (3) residual diving (``_dive``), then (4) the exact
arc-flow MIP (arcflow.py) capped at the incumbent. The best is kept, and
oversupplied items are dropped from patterns to give an exact partition.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from vinpack.io.instances import BinPackingInstance
from vinpack.solvers.backend import build, solve
from vinpack.solvers.bpp import BPPResult, ffd, lower_bound_l2
from vinpack.solvers.kp_dp import kp_dp


def _binary_split(sizes, bounds):
    """Split bounded item types into 0-1 copies of 1, 2, 4, ... units."""
    copy_type, copy_mult = [], []
    for t, (_, u) in enumerate(zip(sizes, bounds, strict=True)):
        k = 1
        while u > 0:
            take = min(k, u)
            copy_type.append(t)
            copy_mult.append(take)
            u -= take
            k *= 2
    return np.array(copy_type), np.array(copy_mult)


def column_generation(
    inst: BinPackingInstance,
    backend: str = "highs",
    max_iters: int = 2000,
    time_limit: float = 60.0,
    int_time_limit: float = 20.0,
    dive_time_limit: float = 10.0,
    arcflow_time_limit: float = 120.0,
) -> BPPResult:
    t0 = time.perf_counter()
    C = int(inst.capacity)
    types, inverse, demand = np.unique(inst.sizes, return_inverse=True, return_counts=True)
    T = len(types)
    bounds = np.minimum(demand, C // types)
    copy_type, copy_mult = _binary_split(types, bounds)
    copy_w = types[copy_type] * copy_mult

    # initial columns: FFD patterns + one homogeneous pattern per type
    start = ffd(inst)
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(a: np.ndarray) -> bool:
        key = a.astype(np.int32).tobytes()
        if key in seen:
            return False
        seen.add(key)
        patterns.append(a.astype(np.int32))
        return True

    for b in start.bins:
        add(np.bincount(inverse[b], minlength=T))
    for t in range(T):
        a = np.zeros(T, dtype=np.int32)
        a[t] = bounds[t]
        add(a)

    lp_val, x_lp, it, history = _cg_loop(patterns, add, demand, copy_type, copy_mult, copy_w, C,
                                         backend, max_iters, t0 + time_limit)
    lp_bound = int(np.ceil(lp_val - 1e-6))
    lb = max(lp_bound, lower_bound_l2(inst.sizes, C))

    # ---- integer phase 1: restricted master IP over generated columns
    P = np.column_stack(patterns).astype(float)
    ip = build(np.ones(P.shape[1]), sp.csr_matrix(P), demand.astype(float), np.inf,
               integer=True, col_ub=np.inf, sense="min")
    ip_sol = solve(ip, backend, time_limit=int_time_limit)
    cands = [np.floor(x_lp + 1e-9).astype(int)]
    if ip_sol.feasible:
        cands.append(np.rint(ip_sol.x).astype(int))

    best_bins = start.bins
    for lam in cands:
        bins = _materialise(inst, patterns, lam, inverse)
        if len(bins) < len(best_bins):
            best_bins = bins
    used_dive = False
    # ---- integer phase 2: residual diving (fix columns, re-price the residual, repeat)
    if len(best_bins) > lb:
        dived = _dive(list(patterns), types, demand, C, backend, max_iters,
                      time.perf_counter() + dive_time_limit)
        if dived is not None:
            bins = _materialise(inst, dived[0], dived[1], inverse)
            if len(bins) < len(best_bins):
                best_bins, used_dive = bins, True
    # ---- integer phase 3: exact arc-flow MIP, bounded by the incumbent
    used_arcflow = False
    if len(best_bins) > lb and arcflow_time_limit > 0:
        from vinpack.solvers.arcflow import arcflow

        af = arcflow(inst, backend, arcflow_time_limit, upper_bound=len(best_bins))
        lb = max(lb, af.lower_bound)
        if af.bins and af.n_bins < len(best_bins):
            best_bins, used_arcflow = af.bins, True
    status = "optimal" if len(best_bins) == lb else "feasible"
    return BPPResult(
        best_bins, len(best_bins), lb, f"colgen-{backend}", time.perf_counter() - t0, status,
        extra={"lp_bound": lp_val, "columns": len(patterns), "iterations": it,
               "lp_history": history, "dive": used_dive,
               "arcflow": used_arcflow},
    )


def _cg_loop(patterns, add, demand, copy_type, copy_mult, copy_w, C, backend, max_iters, deadline):
    """Solve the master LP to optimality for ``demand`` by adding priced-out patterns."""
    T = len(demand)
    history = []
    it = 0
    for it in range(1, max_iters + 1):
        A = sp.csr_matrix(np.column_stack(patterns).astype(float))
        master = build(np.ones(len(patterns)), A, demand.astype(float), np.inf,
                       integer=False, col_ub=np.inf, sense="min")
        sol = solve(master, backend)
        pi = np.maximum(sol.row_duals, 0.0)
        # price: max pi.a  s.t. s.a <= C  (bounded knapsack via 0-1 copies)
        val, xc = kp_dp(pi[copy_type] * copy_mult, copy_w, C)
        history.append(sol.objective)
        if val <= 1 + 1e-9 or time.perf_counter() > deadline:
            break
        a = np.bincount(copy_type, weights=xc * copy_mult, minlength=T).astype(np.int32)
        if not add(a):
            break
    return sol.objective, sol.x, it, history


def _pattern_pool(patterns: list[np.ndarray]):
    seen = {q.astype(np.int32).tobytes() for q in patterns}

    def add(q: np.ndarray) -> bool:
        key = q.astype(np.int32).tobytes()
        if key in seen:
            return False
        seen.add(key)
        patterns.append(q.astype(np.int32))
        return True

    return add


def _dive(patterns, types, demand, C, backend, max_iters, deadline):
    """Residual diving for an exact partition.

    Repeatedly: keep only patterns that fit inside the residual demand, re-solve
    the master LP with pricing restricted to the residual, then fix the column
    with the largest LP value (all columns with lambda >= 1 at once). This avoids
    the oversupply that makes simple rounding lose a bin on perfect-fit instances.
    """
    residual = demand.astype(np.int64).copy()
    share = _pattern_pool(patterns)
    chosen: list[np.ndarray] = []
    counts: list[int] = []
    while residual.sum() > 0:
        if time.perf_counter() > deadline:
            return None
        bounds = np.minimum(residual, C // types)
        local = [q for q in patterns if np.all(q <= residual) and q.any()]
        add = _pattern_pool(local)
        for t in np.flatnonzero(bounds):
            q = np.zeros(len(types), dtype=np.int32)
            q[t] = bounds[t]
            add(q)
        ct, cm = _binary_split(types, bounds)
        _, x, _, _ = _cg_loop(local, add, residual, ct, cm, types[ct] * cm, C, backend,
                              max_iters, deadline)
        take = np.floor(x + 1e-6).astype(int)
        if take.sum() == 0:
            take = np.zeros(len(x), dtype=int)
            take[int(np.argmax(x))] = 1
        for q, k in zip(local, take, strict=True):
            k = int(min(k, np.min(residual[q > 0] // q[q > 0])))
            if k > 0:
                chosen.append(q)
                counts.append(k)
                residual -= q * k
        for q in local:  # share newly priced columns with later residual problems
            share(q)
    return chosen, counts


def _materialise(inst, patterns, lam, inverse) -> list[list[int]]:
    """Turn pattern multiplicities into concrete bins, drop oversupply, FFD the rest."""
    pools: dict[int, list[int]] = {}
    for i in np.argsort(-inst.sizes, kind="stable"):
        pools.setdefault(int(inverse[i]), []).append(int(i))
    bins: list[list[int]] = []
    for p, k in zip(patterns, lam, strict=True):
        for _ in range(int(k)):
            b = []
            for t in np.flatnonzero(p):
                for _ in range(int(p[t])):
                    if pools.get(int(t)):
                        b.append(pools[int(t)].pop())
            if b:
                bins.append(b)
    rest = [i for lst in pools.values() for i in lst]
    if rest:
        sub = BinPackingInstance("residual", inst.sizes[rest], inst.capacity)
        for b in ffd(sub).bins:
            bins.append([rest[i] for i in b])
    return bins
