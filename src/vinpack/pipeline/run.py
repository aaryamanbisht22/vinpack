"""Solve one planning day: L1 allocation -> L2 matching -> L3 loading, with stability.

``solve_day`` is the unit of work the daily orchestrator runs. It is pure:
(scenario, world for day d, yesterday's plan, params) -> today's plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vinpack.io.instances import BinPackingInstance, GAPInstance
from vinpack.pipeline.adapter import Scenario
from vinpack.simulate.scenarios import DayWorld
from vinpack.solvers import gap, mkp
from vinpack.solvers.colgen import column_generation

HOLD_COST = 1_000  # cost of leaving a filled order unmatched (dummy agent); far above any lane


@dataclass(frozen=True)
class DayParams:
    churn_weight: float = 0.5  # lambda as a multiple of the mean order value / mean match cost
    lock_after: int = 3  # consecutive filled days after which an order is "promised"
    lock_penalty: float = 1e5
    backend: str = "highs"
    l1_time: float = 5.0
    l2_time: float = 5.0
    mip_gap: float = 1e-4
    force_fill: tuple[int, ...] = ()  # planner overrides (global order ids)
    extra_supply: tuple[tuple[int, int], ...] = ()  # (pool, units) what-if additions


@dataclass
class DayPlan:
    day: int
    open_orders: np.ndarray
    filled: np.ndarray  # (N,) bool over the global order universe
    agent: np.ndarray  # (N,) agent index, -1 = not matched
    truck: np.ndarray  # (N,) truck index within its agent, -1 = none
    streak: np.ndarray  # (N,) consecutive days filled (promise counter)
    kpis: dict
    pool_capacity: np.ndarray
    agent_capacity: np.ndarray
    events: list[str] = field(default_factory=list)
    l1_duals: np.ndarray | None = None
    l1_reduced_costs: np.ndarray | None = None  # (N,) NaN for closed orders


def solve_day(scn: Scenario, world: DayWorld, prev: DayPlan | None, p: DayParams) -> DayPlan:
    t0 = time.perf_counter()
    N = scn.n_orders
    open_ids = world.open_orders
    pool_cap = world.pool_capacity.copy()
    for pool, units in p.extra_supply:
        pool_cap[pool] += units

    # ------------------------------------------------------------------ L1 allocation
    inst = scn.mkp(open_ids, pool_cap)
    lam1 = p.churn_weight * float(scn.values.mean())
    prev_x = np.full(len(open_ids), np.nan)
    locked = np.zeros(len(open_ids), dtype=bool)
    warm = None
    if prev is not None:
        was_open = np.isin(open_ids, prev.open_orders)
        prev_x[was_open] = prev.filled[open_ids[was_open]].astype(float)
        locked = prev.streak[open_ids] >= p.lock_after
        warm = np.nan_to_num(prev_x, nan=0.0)
        # warm start must be feasible: drop until it fits today's supply
        warm = _fit(inst, warm)
    stab = mkp.Stability(prev_x, lam1, locked, p.lock_penalty) if prev is not None else None
    force = np.flatnonzero(np.isin(open_ids, np.array(p.force_fill, dtype=int)))
    r1 = mkp.solve_mip(inst, p.backend, p.l1_time, p.mip_gap, stab, warm,
                       fix_one=force if len(force) else None)
    if not np.isfinite(r1.value):  # forced orders made it infeasible: report, don't crash
        raise ValueError("L1 infeasible - forced orders exceed today's supply")
    lp = mkp.solve_lp(inst, p.backend)
    filled_ids = open_ids[r1.x == 1]

    # ------------------------------------------------------------------ L2 matching
    A = scn.n_agents
    cost = np.vstack([scn.match_cost[:, filled_ids], np.full((1, len(filled_ids)), HOLD_COST)])
    load = np.vstack([scn.match_load[:, filled_ids], np.zeros((1, len(filled_ids)), np.int64)])
    cap = np.r_[world.agent_capacity, 10**9]
    ginst = GAPInstance(f"day{world.day}:L2", cost, load, cap, "min")
    prev_agent = np.full(len(filled_ids), -1)
    g_locked = np.zeros(len(filled_ids), dtype=bool)
    if prev is not None:
        prev_agent = prev.agent[filled_ids]
        g_locked = prev.streak[filled_ids] >= p.lock_after
    lam2 = p.churn_weight * float(scn.match_cost.mean())
    gstab = gap.GAPStability(prev_agent, lam2, g_locked, p.lock_penalty) if prev is not None else None
    ws = prev_agent.copy() if prev is not None else None
    if ws is not None:
        ws[ws < 0] = A  # new orders start on the hold agent
        if not gap.is_feasible(ginst, ws):
            ws = None
    r2 = gap.solve_mip(ginst, p.backend, p.l2_time, p.mip_gap, gstab, ws)
    assign = r2.assign

    # ------------------------------------------------------------------ L3 loading
    agent = np.full(N, -1)
    truck = np.full(N, -1)
    trucks = trucks_lb = 0
    l3_optimal = 0
    for i in range(A):
        ids = filled_ids[assign == i]
        agent[ids] = i
        if len(ids) == 0:
            continue
        b = BinPackingInstance(f"day{world.day}:{i}", scn.load_size[ids], scn.carrier_capacity)
        res = column_generation(b, p.backend, time_limit=10, dive_time_limit=2,
                                arcflow_time_limit=5)
        for k, items in enumerate(res.bins):
            truck[ids[items]] = k
        trucks += res.n_bins
        trucks_lb += res.lower_bound
        l3_optimal += res.proven_optimal
    held = filled_ids[assign == A]

    # ------------------------------------------------------------------ bookkeeping
    filled = np.zeros(N, dtype=bool)
    filled[filled_ids] = True
    streak = np.zeros(N, dtype=np.int64)
    if prev is not None:
        streak[filled_ids] = prev.streak[filled_ids] + 1
    else:
        streak[filled_ids] = 1
    rc = np.full(N, np.nan)
    rc[open_ids] = lp.reduced_costs

    churn_l1 = churn_l2 = broken = 0
    if prev is not None:
        both = np.intersect1d(open_ids, prev.open_orders)
        churn_l1 = int((filled[both] != prev.filled[both]).sum())
        matched_both = both[(prev.agent[both] >= 0) & (agent[both] >= 0)]
        churn_l2 = int((agent[matched_both] != prev.agent[matched_both]).sum())
        promised = both[prev.streak[both] >= p.lock_after]
        broken = int(((~filled[promised]) | (agent[promised] != prev.agent[promised])).sum())
    match_cost = float(scn.match_cost[agent[agent >= 0], np.flatnonzero(agent >= 0)].sum())
    kpis = {
        "day": world.day,
        "open": int(len(open_ids)),
        "filled": int(len(filled_ids)),
        "value": float(r1.value),
        "l1_lp_bound": float(lp.value),  # upper bound on value with no stability terms
        "l1_gap": float(abs(r1.bound - r1.extra["objective_with_churn"]) / max(1, r1.value)),
        "l1_status": r1.status,
        "match_cost": match_cost,
        "l2_status": r2.status,
        "held": int(len(held)),
        "trucks": int(trucks),
        "trucks_lb": int(trucks_lb),
        "l3_proven": int(l3_optimal),
        "churn_l1": churn_l1,
        "churn_l2": churn_l2,
        "broken_promises": broken,
        "runtime": time.perf_counter() - t0,
    }
    return DayPlan(world.day, open_ids, filled, agent, truck, streak, kpis, pool_cap,
                   world.agent_capacity, list(world.events), lp.duals, rc)


def _fit(inst, x: np.ndarray) -> np.ndarray:
    """Drop lowest-value orders from ``x`` until it satisfies every pool."""
    x = x.copy()
    used = inst.resources @ x
    for j in np.argsort(inst.profits):
        if np.all(used <= inst.capacities):
            break
        if x[j]:
            x[j] = 0
            used -= inst.resources[:, j]
    return x
