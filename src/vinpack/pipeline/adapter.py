"""Compose three academic benchmark instances into one allocation -> matching -> loading scenario.

Every number in the scenario comes from a published benchmark file; the adapter
only decides which benchmark row plays which business role:

* L1  mknapcb5 (10 pools x 250 orders): profits = order contribution, the 10
      constraint rows = scarce supply pools, b = weekly supply.
* L2  OR-Library GAP type C (10 agents): columns of gapc 10x200-4 and the first
      50 columns of gapc 10x100-3 are concatenated to cover all 250 orders.
      Type C capacities are defined as b_i = 0.8 * sum_j r_ij / m, a formula
      that is additive over jobs, so it is recomputed exactly on the
      concatenated columns, then scaled by the expected fill rate (the share of
      orders L1 fills on the full pool) because only filled orders are matched.
* L3  Falkenauer u250_00: 250 item sizes -> the load units each order occupies
      on a carrier of capacity 150.

The composed scenario is NOT a published benchmark; per-layer results on the
unmodified instances are reported separately by ``benchmarks/run_all.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from vinpack.io.instances import MKPInstance
from vinpack.io.orlib import read_binpack, read_gap, read_mknap

DATA = Path(__file__).resolve().parents[3] / "data" / "raw"

POOL_NAMES = [
    "Long-range battery packs",
    "Standard-range battery packs",
    "Dual-motor drive units",
    "Performance drive units",
    "Paint shop: premium colors",
    "Interior kits: white",
    "20-inch wheel sets",
    "Tow-hitch kits",
    "Heat-pump modules",
    "Final-line hours",
]
AGENT_NAMES = [
    "Los Angeles DC",
    "Bay Area DC",
    "Seattle DC",
    "Phoenix DC",
    "Denver DC",
    "Dallas DC",
    "Chicago DC",
    "Atlanta DC",
    "New Jersey DC",
    "Orlando DC",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    values: np.ndarray  # (N,) order contribution  (L1 profit)
    pool_use: np.ndarray  # (P, N) supply consumed by each order
    pool_capacity: np.ndarray  # (P,)
    match_cost: np.ndarray  # (A, N) cost of fulfilling order j via agent i (minimise)
    match_load: np.ndarray  # (A, N) handling capacity consumed
    agent_capacity: np.ndarray  # (A,)
    load_size: np.ndarray  # (N,) carrier load units
    carrier_capacity: int
    pool_names: tuple[str, ...]
    agent_names: tuple[str, ...]
    sources: dict

    @property
    def n_orders(self) -> int:
        return len(self.values)

    @property
    def n_pools(self) -> int:
        return len(self.pool_capacity)

    @property
    def n_agents(self) -> int:
        return len(self.agent_capacity)

    def mkp(self, orders: np.ndarray, capacity: np.ndarray | None = None) -> MKPInstance:
        return MKPInstance(
            name=f"{self.name}:L1",
            profits=self.values[orders],
            resources=self.pool_use[:, orders],
            capacities=(self.pool_capacity if capacity is None else capacity).astype(np.int64),
        )


@lru_cache(maxsize=4)
def load_default(data_dir: str | None = None) -> Scenario:
    d = Path(data_dir) if data_dir else DATA
    mk = read_mknap(d / "mknapcb5.txt", d / "mkcbres.txt")[20]  # 10.250-20, tightness 0.75
    g_big = read_gap(d / "gapc.txt")[3]  # 10x200-4
    g_small = read_gap(d / "gapc.txt")[2]  # 10x100-3
    bp = next(b for b in read_binpack(d / "binpack2.txt") if b.name == "u250_00")
    n = mk.n
    extra = n - g_big.n
    cost = np.hstack([g_big.costs, g_small.costs[:, :extra]])
    load = np.hstack([g_big.resources, g_small.resources[:, :extra]])
    m = cost.shape[0]
    cap_full = np.floor(0.8 * load.sum(axis=1) / m)

    # expected fill rate: share of orders the L1 dual-greedy heuristic fills on the full pool
    from vinpack.solvers.mkp import greedy_dual

    fill = greedy_dual(mk).x.mean()
    agent_cap = np.floor(cap_full * fill).astype(np.int64)
    return Scenario(
        name="default",
        values=mk.profits.astype(np.int64),
        pool_use=mk.resources.astype(np.int64),
        pool_capacity=mk.capacities.astype(np.int64),
        match_cost=cost.astype(np.int64),
        match_load=load.astype(np.int64),
        agent_capacity=agent_cap,
        load_size=bp.sizes[:n].astype(np.int64),
        carrier_capacity=int(bp.capacity),
        pool_names=tuple(POOL_NAMES[: mk.m]),
        agent_names=tuple(AGENT_NAMES[:m]),
        sources={
            "L1": f"OR-Library {mk.name} (Chu & Beasley 1998)",
            "L2": f"OR-Library {g_big.name} + first {extra} jobs of {g_small.name}",
            "L3": f"OR-Library binpack2 {bp.name} (Falkenauer 1996)",
            "fill_rate": float(fill),
        },
    )
