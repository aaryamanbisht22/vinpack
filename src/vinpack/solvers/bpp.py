"""L3 loading: one-dimensional bin packing (VINs onto car-carrier trucks / railcars).

Methods:
* ``ffd`` / ``bfd``          first-fit / best-fit decreasing heuristics.
* ``lower_bound_l2``         Martello-Toth L2 lower bound.
* ``compact_mip``            Kantorovich assignment model (x_ik, y_k) with symmetry
                             breaking. Kept deliberately: its LP bound is the trivial
                             ceil(sum/C), which is why nobody solves BPP this way.
* ``column_generation``      see colgen.py (Gilmore-Gomory, priced by the L0 kernel).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from vinpack.io.instances import BinPackingInstance
from vinpack.solvers.backend import build, solve


@dataclass
class BPPResult:
    bins: list[list[int]]  # item indices per bin
    n_bins: int
    lower_bound: int
    method: str
    runtime: float
    status: str = "feasible"
    extra: dict = field(default_factory=dict)

    @property
    def proven_optimal(self) -> bool:
        return self.n_bins == self.lower_bound


def is_feasible(inst: BinPackingInstance, bins: list[list[int]]) -> bool:
    seen = sorted(i for b in bins for i in b)
    return seen == list(range(inst.n)) and all(inst.sizes[b].sum() <= inst.capacity for b in bins)


def lower_bound_l1(sizes, cap) -> int:
    return int(np.ceil(np.sum(sizes) / cap))


def lower_bound_l2(sizes, cap) -> int:
    """Martello & Toth (1990) L2 bound: max over alpha in [0, C/2]."""
    s = np.sort(np.asarray(sizes))[::-1]
    best = lower_bound_l1(s, cap)
    for alpha in np.unique(np.r_[0, s[s <= cap / 2]]):
        big = s[s > cap - alpha]
        mid = s[(s > cap / 2) & (s <= cap - alpha)]
        small = s[(s >= alpha) & (s <= cap / 2)]
        free = len(mid) * cap - mid.sum()
        extra = max(0.0, np.ceil((small.sum() - free) / cap))
        best = max(best, int(len(big) + len(mid) + extra))
    return best


def _fit_decreasing(inst: BinPackingInstance, best_fit: bool) -> BPPResult:
    t0 = time.perf_counter()
    order = np.argsort(-inst.sizes, kind="stable")
    loads: list[int] = []
    bins: list[list[int]] = []
    for i in order:
        s = inst.sizes[i]
        slack = np.array([inst.capacity - L - s for L in loads]) if loads else np.array([])
        ok = np.flatnonzero(slack >= 0)
        if len(ok) == 0:
            loads.append(int(s))
            bins.append([int(i)])
            continue
        k = ok[np.argmin(slack[ok])] if best_fit else ok[0]
        loads[k] += int(s)
        bins[k].append(int(i))
    lb = lower_bound_l2(inst.sizes, inst.capacity)
    return BPPResult(bins, len(bins), lb, "bfd" if best_fit else "ffd", time.perf_counter() - t0)


def ffd(inst: BinPackingInstance) -> BPPResult:
    return _fit_decreasing(inst, best_fit=False)


def bfd(inst: BinPackingInstance) -> BPPResult:
    return _fit_decreasing(inst, best_fit=True)


def compact_mip(inst: BinPackingInstance, backend: str = "highs", time_limit: float = 30.0) -> BPPResult:
    """x_ik = item i in bin k, y_k = bin k used; K = FFD upper bound."""
    t0 = time.perf_counter()
    ub = ffd(inst)
    K, n, C = ub.n_bins, inst.n, inst.capacity
    nx = n * K
    c = np.r_[np.zeros(nx), np.ones(K)]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for i in range(n):  # each item in one bin
        rows += [r] * K
        cols += [i * K + k for k in range(K)]
        vals += [1.0] * K
        lo.append(1)
        hi.append(1)
        r += 1
    for k in range(K):  # capacity linked to y_k
        rows += [r] * (n + 1)
        cols += [i * K + k for i in range(n)] + [nx + k]
        vals += list(inst.sizes.astype(float)) + [-float(C)]
        lo.append(-np.inf)
        hi.append(0)
        r += 1
    for k in range(K - 1):  # symmetry breaking y_k >= y_{k+1}
        rows += [r, r]
        cols += [nx + k, nx + k + 1]
        vals += [1.0, -1.0]
        lo.append(0)
        hi.append(np.inf)
        r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nx + K))
    model = build(c, A, lo, hi, integer=True, sense="min")
    ws = np.zeros(nx + K)
    for k, b in enumerate(ub.bins):
        ws[[i * K + k for i in b]] = 1
        ws[nx + k] = 1
    sol = solve(model, backend, time_limit, warm_start=ws)
    lb = max(lower_bound_l2(inst.sizes, C), int(np.ceil(sol.bound - 1e-6)) if np.isfinite(sol.bound) else 0)
    if not sol.feasible:
        return BPPResult(ub.bins, ub.n_bins, lb, f"compact-{backend}", time.perf_counter() - t0, sol.status)
    X = np.rint(sol.x[:nx]).reshape(n, K)
    bins = [list(np.flatnonzero(X[:, k])) for k in range(K) if X[:, k].any()]
    return BPPResult(bins, len(bins), lb, f"compact-{backend}", time.perf_counter() - t0, sol.status)
