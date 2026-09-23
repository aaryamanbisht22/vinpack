"""Lagrangian relaxations whose subproblems are 0-1 knapsacks, solved by the L0 kernel.

MKP (L1): relax every supply pool except one ("kept" pool k) with multipliers
lambda >= 0. What is left is a single 0-1 knapsack on pool k with profits
p_j - sum_{i != k} lambda_i r_ij, solved exactly by ``kp_dp``. Because the
subproblem lacks the integrality property, this bound can be strictly tighter
than the LP bound.

GAP (L2): relax the "each job to exactly one agent" rows with multipliers u_j
(free sign). The relaxation decomposes into one 0-1 knapsack per agent - the
classic Ross-Soland / Fisher-Jaikumar-Van Wassenhove bound.

Both use subgradient optimisation with a Polyak step against the best feasible
value found by a Lagrangian repair heuristic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vinpack.io.instances import GAPInstance, MKPInstance
from vinpack.solvers.kp_dp import kp_dp


@dataclass
class LagrangianResult:
    x: np.ndarray
    value: float  # best feasible value found
    bound: float  # best Lagrangian bound
    iterations: int
    runtime: float
    history: list[tuple[float, float]] = field(default_factory=list)  # (bound, best)
    method: str = "lagrangian"

    @property
    def gap(self) -> float:
        return abs(self.bound - self.value) / max(1.0, abs(self.value))


# ------------------------------------------------------------------------------ MKP
def mkp_lagrangian(
    inst: MKPInstance,
    iters: int = 150,
    keep: int | None = None,
    lp_duals: np.ndarray | None = None,
    time_limit: float = 60.0,
) -> LagrangianResult:
    from vinpack.solvers.mkp import greedy_dual, solve_lp

    t0 = time.perf_counter()
    R, b, p = inst.resources.astype(float), inst.capacities.astype(float), inst.profits.astype(float)
    if lp_duals is None:
        lp_duals = solve_lp(inst).duals
    if keep is None:
        keep = int(np.argmax(lp_duals * b))  # the pool carrying most of the LP value
    others = np.array([i for i in range(inst.m) if i != keep])
    lam = lp_duals[others].copy()  # LP duals are an excellent starting point

    start = greedy_dual(inst, lp_duals)
    best_x, best_val = start.x.copy(), start.value
    best_bound = np.inf
    history = []
    step_scale = 2.0
    stall = 0
    wk = inst.resources[keep].astype(np.int64)
    for it in range(1, iters + 1):
        red = p - lam @ R[others]
        sub_val, x = kp_dp(red, wk, int(b[keep]))
        bound = sub_val + lam @ b[others]
        if bound < best_bound - 1e-9:
            best_bound, stall = bound, 0
        else:
            stall += 1
            if stall >= 10:
                step_scale, stall = step_scale / 2, 0
        # Lagrangian heuristic: repair x to feasibility by dropping worst-ratio orders, refill.
        cand = _repair_mkp(inst, x.astype(np.int8), lp_duals)
        val = float(p @ cand)
        if val > best_val:
            best_val, best_x = val, cand
        history.append((float(best_bound), float(best_val)))
        g = b[others] - R[others] @ x
        norm = float(g @ g)
        if norm < 1e-12 or best_bound - best_val < 1 - 1e-9 or step_scale < 1e-4:
            break
        if time.perf_counter() - t0 > time_limit:
            break
        step = step_scale * (bound - best_val) / norm
        lam = np.maximum(0.0, lam - step * g)
    # integer profits: floor the bound
    best_bound = np.floor(best_bound + 1e-6)
    return LagrangianResult(best_x, best_val, float(best_bound), it,
                            time.perf_counter() - t0, history, "lagrangian-kp")


def _repair_mkp(inst: MKPInstance, x: np.ndarray, duals: np.ndarray) -> np.ndarray:
    R, b, p = inst.resources, inst.capacities, inst.profits
    weight = duals @ R + 1e-6 * (R / np.maximum(b[:, None], 1)).sum(0)
    ratio = p / weight
    x = x.copy()
    used = R @ x
    for j in np.argsort(ratio):  # drop worst first
        if np.all(used <= b):
            break
        if x[j]:
            x[j] = 0
            used -= R[:, j]
    for j in np.argsort(-ratio):  # add best first
        if not x[j] and np.all(used + R[:, j] <= b):
            x[j] = 1
            used += R[:, j]
    return x


# ------------------------------------------------------------------------------ GAP
def gap_lagrangian(inst: GAPInstance, iters: int = 300, time_limit: float = 60.0) -> LagrangianResult:
    """Works in maximisation form internally; min instances are negated and flipped back."""
    t0 = time.perf_counter()
    sign = 1.0 if inst.sense == "max" else -1.0
    C = sign * inst.costs.astype(float)  # maximise C
    Rm = inst.resources.astype(np.int64)
    cap = inst.capacities.astype(np.int64)
    m, n = C.shape

    u = C.max(axis=0) if sign > 0 else np.sort(C, axis=0)[-2]  # start: second best value
    best_assign, best_val = None, -np.inf
    best_bound = np.inf
    history = []
    step_scale, stall = 2.0, 0
    it = 0
    for it in range(1, iters + 1):
        X = np.zeros((m, n), dtype=np.int8)
        total = u.sum()
        for i in range(m):
            v, xi = kp_dp(C[i] - u, Rm[i], int(cap[i]))
            X[i] = xi
            total += v
        if total < best_bound - 1e-9:
            best_bound, stall = total, 0
        else:
            stall += 1
            if stall >= 15:
                step_scale, stall = step_scale / 2, 0
        assign = _repair_gap(C, Rm, cap, X)
        if assign is not None:
            val = float(C[assign, np.arange(n)].sum())
            if val > best_val:
                best_val, best_assign = val, assign
        history.append((sign * float(best_bound), sign * float(best_val) if np.isfinite(best_val) else np.nan))
        g = 1 - X.sum(axis=0)  # subgradient of the dualised equality rows
        norm = float(g @ g)
        if norm == 0:  # the relaxed solution is feasible, hence optimal
            break
        if np.isfinite(best_val) and best_bound - best_val < 1 - 1e-9:
            break
        if step_scale < 1e-4 or time.perf_counter() - t0 > time_limit:
            break
        target = best_val if np.isfinite(best_val) else best_bound - 0.05 * abs(best_bound)
        step = step_scale * (total - target) / norm
        u = u - step * g  # minimise L(u): move against the subgradient
    bound = np.floor(best_bound + 1e-6)
    return LagrangianResult(
        x=best_assign if best_assign is not None else np.full(n, -1),
        value=sign * best_val if np.isfinite(best_val) else np.nan,
        bound=sign * bound,
        iterations=it,
        runtime=time.perf_counter() - t0,
        history=history,
        method="lagrangian-kp",
    )


def _repair_gap(C, R, cap, X) -> np.ndarray | None:
    """Turn a relaxed solution into a feasible assignment (or None)."""
    m, n = C.shape
    assign = np.full(n, -1)
    load = np.zeros(m, dtype=np.int64)
    # 1) keep single assignments; for multiply-assigned jobs keep the best-value agent
    for j in np.argsort(-C.max(axis=0)):
        agents = np.flatnonzero(X[:, j])
        if len(agents):
            i = agents[np.argmax(C[agents, j])]
            if load[i] + R[i, j] <= cap[i]:
                assign[j], load[i] = i, load[i] + R[i, j]
    # 2) place unassigned jobs by regret (difference between best and second-best feasible value)
    todo = list(np.flatnonzero(assign < 0))
    while todo:
        best_j, best_i, best_regret = None, None, -np.inf
        for j in todo:
            feas = np.flatnonzero(load + R[:, j] <= cap)
            if len(feas) == 0:
                return _shift_to_make_room(C, R, cap, assign, load, todo)
            vals = np.sort(C[feas, j])[::-1]
            regret = vals[0] - (vals[1] if len(vals) > 1 else -1e9)
            if regret > best_regret:
                best_regret, best_j, best_i = regret, j, feas[np.argmax(C[feas, j])]
        assign[best_j] = best_i
        load[best_i] += R[best_i, best_j]
        todo.remove(best_j)
    return _local_search_gap(C, R, cap, assign, load)


def _shift_to_make_room(C, R, cap, assign, load, todo):
    """Last resort: try to move an assigned job to free room for an unplaceable one."""
    for j in list(todo):
        placed = False
        for i in np.argsort(-C[:, j]):
            need = load[i] + R[i, j] - cap[i]
            for k in np.flatnonzero(assign == i):
                if R[i, k] < need:
                    continue
                alt = np.flatnonzero((load + R[:, k] <= cap) & (np.arange(len(cap)) != i))
                if len(alt):
                    i2 = alt[np.argmax(C[alt, k])]
                    assign[k] = i2
                    load[i] -= R[i, k]
                    load[i2] += R[i2, k]
                    assign[j] = i
                    load[i] += R[i, j]
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
        todo.remove(j)
    return _local_search_gap(C, R, cap, assign, load)


def _local_search_gap(C, R, cap, assign, load, rounds: int = 20):
    m, n = C.shape
    cols = np.arange(n)
    for _ in range(rounds):
        improved = False
        # shift moves
        for j in range(n):
            i = assign[j]
            feas = (load + R[:, j] <= cap)
            feas[i] = False
            gain = np.where(feas, C[:, j] - C[i, j], -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > 1e-9:
                load[i] -= R[i, j]
                load[k] += R[k, j]
                assign[j] = k
                improved = True
        # swap moves
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                a, b = assign[j1], assign[j2]
                if a == b:
                    continue
                d = C[b, j1] + C[a, j2] - C[a, j1] - C[b, j2]
                if d <= 1e-9:
                    continue
                if (load[a] - R[a, j1] + R[a, j2] <= cap[a]
                        and load[b] - R[b, j2] + R[b, j1] <= cap[b]):
                    load[a] += R[a, j2] - R[a, j1]
                    load[b] += R[b, j1] - R[b, j2]
                    assign[j1], assign[j2] = b, a
                    improved = True
        if not improved:
            break
    assert np.all(load <= cap) and np.all(assign >= 0), (load, cap)
    _ = cols
    return assign
