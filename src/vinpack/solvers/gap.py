"""L2 matching: the generalized assignment problem (GAP).

    opt  sum_ij c_ij y_ij
    s.t. sum_i y_ij = 1            every filled order is matched to exactly one agent
         sum_j r_ij y_ij <= b_i    agent (delivery center / outbound lane) capacity
         y_ij in {0, 1}

Variables are laid out agent-major: column k = i * n + j.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from vinpack.io.instances import GAPInstance
from vinpack.solvers.backend import LinearModel, build, solve


@dataclass
class GAPStability:
    """Churn penalty for re-matching an order that was matched to a different agent yesterday.

    ``prev[j]`` is yesterday's agent for order j, or -1 if it was not matched.
    """

    prev: np.ndarray
    weight: float = 0.0
    locked: np.ndarray | None = None
    lock_penalty: float = 1e6


@dataclass
class GAPResult:
    assign: np.ndarray  # agent index per job, -1 if unassigned
    value: float  # pure cost/value, without churn terms
    bound: float
    method: str
    runtime: float
    status: str = "optimal"
    extra: dict = field(default_factory=dict)


def value_of(inst: GAPInstance, assign: np.ndarray) -> float:
    return float(inst.costs[assign, np.arange(inst.n)].sum())


def is_feasible(inst: GAPInstance, assign: np.ndarray) -> bool:
    if np.any(assign < 0):
        return False
    load = np.zeros(inst.m)
    np.add.at(load, assign, inst.resources[assign, np.arange(inst.n)])
    return bool(np.all(load <= inst.capacities + 1e-9))


def build_model(
    inst: GAPInstance,
    stability: GAPStability | None = None,
    forbid: list[tuple[int, int]] | None = None,
    force: list[tuple[int, int]] | None = None,
) -> LinearModel:
    m, n = inst.m, inst.n
    c = inst.costs.astype(float).ravel().copy()
    maximize = inst.sense == "max"
    if stability is not None:
        # Moving job j away from yesterday's agent costs `weight` (+ lock penalty if promised).
        # Encode as a bonus on keeping y[prev_j, j] = 1: equivalent up to a constant.
        for j in np.flatnonzero(stability.prev >= 0):
            i = int(stability.prev[j])
            pen = stability.weight
            if stability.locked is not None and stability.locked[j]:
                pen += stability.lock_penalty
            c[i * n + j] += pen if maximize else -pen
    rows, cols, vals = [], [], []
    for j in range(n):  # assignment rows
        rows += [j] * m
        cols += [i * n + j for i in range(m)]
        vals += [1.0] * m
    for i in range(m):  # capacity rows
        rows += [n + i] * n
        cols += [i * n + j for j in range(n)]
        vals += list(inst.resources[i].astype(float))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n + m, m * n))
    row_lb = np.r_[np.ones(n), np.full(m, -np.inf)]
    row_ub = np.r_[np.ones(n), inst.capacities.astype(float)]
    names = [f"assign_{j}" for j in range(n)] + [f"capacity_{i}" for i in range(m)]
    model = build(c, A, row_lb, row_ub, integer=True, sense=inst.sense, row_names=names)
    # prune pairs that can never fit
    model.col_ub[(inst.resources > inst.capacities[:, None]).ravel()] = 0
    for i, j in forbid or []:
        model.col_ub[i * n + j] = 0
    for i, j in force or []:
        model.col_lb[i * n + j] = 1
    return model


def solve_mip(
    inst: GAPInstance,
    backend: str = "highs",
    time_limit: float = 30.0,
    mip_gap: float = 1e-6,
    stability: GAPStability | None = None,
    warm_start: np.ndarray | None = None,
    forbid=None,
    force=None,
) -> GAPResult:
    model = build_model(inst, stability, forbid, force)
    ws = None
    if warm_start is not None and np.all(warm_start >= 0):
        ws = np.zeros(inst.m * inst.n)
        ws[warm_start * inst.n + np.arange(inst.n)] = 1
    sol = solve(model, backend, time_limit, mip_gap, warm_start=ws)
    if not sol.feasible:
        return GAPResult(np.full(inst.n, -1), np.nan, np.nan, f"mip-{backend}", sol.runtime,
                         sol.status)
    Y = np.rint(sol.x).reshape(inst.m, inst.n)
    assign = Y.argmax(axis=0)
    return GAPResult(assign, value_of(inst, assign), sol.bound, f"mip-{backend}", sol.runtime,
                     sol.status, extra={"objective_with_churn": sol.objective})


def solve_lp(inst: GAPInstance, backend: str = "highs"):
    sol = solve(build_model(inst), backend, relax=True)
    return sol


def greedy_regret(inst: GAPInstance) -> GAPResult:
    """Martello-Toth style regret heuristic followed by shift/swap local search."""
    import time

    from vinpack.solvers.lagrangian import _repair_gap

    t0 = time.perf_counter()
    sign = 1.0 if inst.sense == "max" else -1.0
    C = sign * inst.costs.astype(float)
    empty = np.zeros((inst.m, inst.n), dtype=np.int8)
    assign = _repair_gap(C, inst.resources.astype(np.int64), inst.capacities.astype(np.int64), empty)
    if assign is None:
        return GAPResult(np.full(inst.n, -1), np.nan, np.nan, "greedy-regret",
                         time.perf_counter() - t0, "infeasible")
    return GAPResult(assign, value_of(inst, assign), np.nan, "greedy-regret",
                     time.perf_counter() - t0)
