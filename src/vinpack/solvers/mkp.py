"""L1 allocation: the multidimensional knapsack problem (MKP).

    max  sum_j p_j x_j
    s.t. sum_j r_ij x_j <= b_i      for every scarce supply pool i
         x_j in {0, 1}              order j is filled this cycle or not

Methods:
* ``solve_mip``     exact MIP on HiGHS or Gurobi (with optional stability terms).
* ``solve_lp``      LP relaxation, returning pool duals and order reduced costs.
* ``greedy_dual``   Chu-Beasley style heuristic: rank orders by p_j / (u . r_j)
                    with u = LP duals, drop-then-add repair, then 1-swap search.
The Lagrangian method lives in ``lagrangian.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from vinpack.io.instances import MKPInstance
from vinpack.solvers.backend import LinearModel, Solution, build, solve


@dataclass
class Stability:
    """Churn penalty against yesterday's plan.

    ``prev`` is yesterday's 0/1 decision (NaN where the order did not exist).
    Changing a decision costs ``weight``; ``locked`` orders are promised, and
    un-filling them costs ``lock_penalty`` (soft, so a supply shock can still
    break a promise instead of making the model infeasible).
    """

    prev: np.ndarray
    weight: float = 0.0
    locked: np.ndarray | None = None
    lock_penalty: float = 1e6


@dataclass
class MKPResult:
    x: np.ndarray
    value: float  # sum p_j x_j (pure business value, without churn terms)
    bound: float
    method: str
    runtime: float
    status: str = "optimal"
    extra: dict = field(default_factory=dict)


def value_of(inst: MKPInstance, x: np.ndarray) -> float:
    return float(inst.profits @ x)


def is_feasible(inst: MKPInstance, x: np.ndarray) -> bool:
    return bool(np.all(inst.resources @ x <= inst.capacities + 1e-9))


def build_model(
    inst: MKPInstance,
    stability: Stability | None = None,
    fix_one: np.ndarray | None = None,
    fix_zero: np.ndarray | None = None,
) -> LinearModel:
    c = inst.profits.astype(float).copy()
    offset = 0.0
    if stability is not None:
        # |x - prev| is linear for binary x: x if prev == 0, (1 - x) if prev == 1.
        prev = stability.prev
        known = ~np.isnan(prev)
        pen = np.zeros_like(c)
        pen[known] = stability.weight
        if stability.locked is not None:
            pen[stability.locked & known] += stability.lock_penalty
        was_on = known & (prev == 1)
        was_off = known & (prev == 0)
        c[was_off] -= pen[was_off]
        c[was_on] += pen[was_on]
        offset = -float(pen[was_on].sum())
    col_lb = np.zeros(inst.n)
    col_ub = np.ones(inst.n)
    if fix_one is not None:
        col_lb[fix_one] = 1
    if fix_zero is not None:
        col_ub[fix_zero] = 0
    names = [f"pool_{i}" for i in range(inst.m)]
    m = build(c, sp.csr_matrix(inst.resources.astype(float)), -np.inf, inst.capacities,
              integer=True, sense="max", row_names=names, obj_offset=offset)
    m.col_lb, m.col_ub = col_lb, col_ub
    return m


def solve_mip(
    inst: MKPInstance,
    backend: str = "highs",
    time_limit: float = 30.0,
    mip_gap: float = 1e-6,
    stability: Stability | None = None,
    warm_start: np.ndarray | None = None,
    fix_one: np.ndarray | None = None,
    fix_zero: np.ndarray | None = None,
) -> MKPResult:
    model = build_model(inst, stability, fix_one, fix_zero)
    sol: Solution = solve(model, backend, time_limit, mip_gap, warm_start=warm_start)
    if not sol.feasible:
        return MKPResult(np.zeros(inst.n, np.int8), np.nan, np.nan, f"mip-{backend}",
                         sol.runtime, sol.status)
    x = np.rint(sol.x).astype(np.int8)
    return MKPResult(
        x, value_of(inst, x), sol.bound, f"mip-{backend}", sol.runtime, sol.status,
        extra={"objective_with_churn": sol.objective, "nodes": sol.extra.get("nodes")},
    )


@dataclass
class LPInfo:
    value: float
    x: np.ndarray
    duals: np.ndarray  # marginal value of one more unit of each pool (>= 0)
    reduced_costs: np.ndarray  # p_j - duals . r_j


def solve_lp(inst: MKPInstance, backend: str = "highs", fix_one=None, fix_zero=None) -> LPInfo:
    sol = solve(build_model(inst, fix_one=fix_one, fix_zero=fix_zero), backend, relax=True)
    duals = np.maximum(sol.row_duals, 0.0)
    rc = inst.profits - duals @ inst.resources
    return LPInfo(sol.objective, sol.x, duals, rc)


def greedy_dual(inst: MKPInstance, duals: np.ndarray | None = None, swaps: bool = True) -> MKPResult:
    t0 = time.perf_counter()
    if duals is None:
        duals = solve_lp(inst).duals
    weight = duals @ inst.resources
    # guard against zero-dual pools: fall back to normalised resource use
    weight = weight + 1e-6 * (inst.resources / np.maximum(inst.capacities[:, None], 1)).sum(0)
    ratio = inst.profits / weight
    order = np.argsort(-ratio, kind="stable")
    x = np.zeros(inst.n, dtype=np.int8)
    used = np.zeros(inst.m)
    for j in order:
        if np.all(used + inst.resources[:, j] <= inst.capacities):
            x[j] = 1
            used += inst.resources[:, j]
    if swaps:
        x = _one_swap(inst, x, order)
    return MKPResult(x, value_of(inst, x), np.nan, "greedy-dual", time.perf_counter() - t0)


def _one_swap(inst: MKPInstance, x: np.ndarray, order: np.ndarray, max_rounds: int = 50):
    """First-improvement search: swap one filled order for one unfilled one, then refill."""
    R, b, p = inst.resources, inst.capacities, inst.profits
    x = x.copy()
    for _ in range(max_rounds):
        used = R @ x
        improved = False
        ins = np.flatnonzero(x == 1)
        outs = np.flatnonzero(x == 0)
        slack = b - used
        for j_out in outs:
            # candidates to remove so that j_out fits and value improves
            need = R[:, j_out] - slack
            gain = p[j_out] - p[ins]
            ok = np.all(R[:, ins] >= need[:, None], axis=0) & (gain > 0)
            if ok.any():
                k = ins[np.argmax(np.where(ok, gain, -np.inf))]
                x[k], x[j_out] = 0, 1
                improved = True
                break
        # greedy refill
        used = R @ x
        for j in order:
            if x[j] == 0 and np.all(used + R[:, j] <= b):
                x[j] = 1
                used += R[:, j]
        if not improved:
            break
    return x
