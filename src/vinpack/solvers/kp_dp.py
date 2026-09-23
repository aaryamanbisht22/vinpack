"""L0 kernel: exact 0-1 knapsack solvers.

Two exact algorithms, both Numba-compiled:

* ``kp_dp``  - dynamic programming over capacity, O(n*C) time. Profits may be
  floats, which is what column-generation pricing needs (profits = LP duals).
* ``kp_bb``  - Horowitz-Sahni depth-first branch and bound with the Dantzig
  (LP) bound, items sorted by profit/weight. Independent of C, so it handles
  large-capacity instances where the DP table would not fit in memory.

``solve_knapsack`` picks between them. Both are validated against Pisinger's
published optimal solutions in tests/test_knapsack.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit

from vinpack.io.instances import KnapsackInstance

DP_CELL_LIMIT = 400_000_000  # n * (C + 1) bytes for the reconstruction table


@njit(cache=True)
def _dp(profits, weights, cap):
    n = profits.shape[0]
    best = np.zeros(cap + 1, dtype=np.float64)
    keep = np.zeros((n, cap + 1), dtype=np.uint8)
    for i in range(n):
        w = weights[i]
        p = profits[i]
        if w > cap or p <= 0.0:
            continue
        for c in range(cap, w - 1, -1):
            cand = best[c - w] + p
            if cand > best[c] + 1e-12:
                best[c] = cand
                keep[i, c] = 1
    x = np.zeros(n, dtype=np.int8)
    c = cap
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            x[i] = 1
            c -= weights[i]
    return best[cap], x


@njit(cache=True)
def _bb(p, w, cap, node_limit):
    """Depth-first B&B on items already sorted by decreasing p/w."""
    n = p.shape[0]
    x = np.zeros(n, dtype=np.int8)
    best_x = np.zeros(n, dtype=np.int8)
    best = 0.0
    nodes = 0
    # iterative DFS: at depth i decide x[i]; try 1 first
    i = 0
    cur_p = 0.0
    cur_w = 0
    choice = np.full(n + 1, -1, dtype=np.int8)  # -1 = not visited, 1 = tried in, 0 = tried out
    while True:
        nodes += 1
        if nodes > node_limit:
            return best, best_x, nodes, False
        if i == n:
            if cur_p > best:
                best = cur_p
                best_x[:] = x
            # backtrack
            i -= 1
            while i >= 0 and choice[i] == 0:
                choice[i] = -1
                i -= 1
            if i < 0:
                return best, best_x, nodes, True
            # item i was in; take it out and continue
            x[i] = 0
            cur_p -= p[i]
            cur_w -= w[i]
            choice[i] = 0
            i += 1
            continue
        # Dantzig bound from item i on
        rem = cap - cur_w
        ub = cur_p
        j = i
        while j < n and w[j] <= rem:
            rem -= w[j]
            ub += p[j]
            j += 1
        if j < n:
            ub += p[j] * rem / w[j]
        if ub <= best + 1e-9:
            # prune: backtrack
            i -= 1
            while i >= 0 and choice[i] == 0:
                choice[i] = -1
                i -= 1
            if i < 0:
                return best, best_x, nodes, True
            x[i] = 0
            cur_p -= p[i]
            cur_w -= w[i]
            choice[i] = 0
            i += 1
            continue
        if w[i] <= cap - cur_w:
            x[i] = 1
            cur_p += p[i]
            cur_w += w[i]
            choice[i] = 1
        else:
            x[i] = 0
            choice[i] = 0
        i += 1


def kp_dp(profits, weights, capacity: int) -> tuple[float, np.ndarray]:
    p = np.asarray(profits, dtype=np.float64)
    w = np.asarray(weights, dtype=np.int64)
    return _dp(p, w, int(capacity))


def kp_bb(profits, weights, capacity: int, node_limit: int = 50_000_000):
    p = np.asarray(profits, dtype=np.float64)
    w = np.asarray(weights, dtype=np.int64)
    keep = (w <= capacity) & (p > 0)
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(-(p[idx] / w[idx]), kind="stable")]
    val, xs, nodes, proven = _bb(p[order], w[order], int(capacity), node_limit)
    x = np.zeros(len(p), dtype=np.int8)
    x[order] = xs
    return val, x, nodes, proven


@dataclass
class KPResult:
    value: float
    x: np.ndarray
    method: str
    runtime: float
    proven: bool = True


def solve_knapsack(inst: KnapsackInstance, method: str = "auto") -> KPResult:
    t0 = time.perf_counter()
    if method == "auto":
        method = "dp" if inst.n * (inst.capacity + 1) <= DP_CELL_LIMIT else "bb"
    if method == "dp":
        v, x = kp_dp(inst.profits, inst.weights, inst.capacity)
        return KPResult(v, x, "dp", time.perf_counter() - t0)
    if method == "bb":
        v, x, _, proven = kp_bb(inst.profits, inst.weights, inst.capacity)
        return KPResult(v, x, "bb", time.perf_counter() - t0, proven)
    raise ValueError(method)
