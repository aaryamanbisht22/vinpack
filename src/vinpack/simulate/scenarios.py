"""Seeded day-by-day world: which orders are open, and how much supply and DC capacity exist.

The world is a pure function of (scenario, seed, day), so any day can be rebuilt
exactly - the explainer relies on this to re-run what-ifs for a past day.

Dynamics (all deterministic given the seed):
* 60% of orders exist on day 0; the rest arrive uniformly over the horizon.
* each open order cancels with probability ``cancel_rate`` per day.
* supply pools follow a clipped random walk around their benchmark capacity,
  scaled by the share of orders that have arrived (supply is planned for the
  full week's demand, so early days are loose and scarcity grows).
* one DC suffers a capacity shock (default -35% for 3 days) mid-horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vinpack.pipeline.adapter import Scenario


@dataclass(frozen=True)
class WorldConfig:
    days: int = 14
    seed: int = 7
    initial_share: float = 0.6
    cancel_rate: float = 0.01
    supply_walk: float = 0.03
    shock_day: int | None = None  # default: days // 2
    shock_agent: int = 6  # Chicago DC
    shock_size: float = 0.35
    shock_len: int = 3


@dataclass(frozen=True)
class DayWorld:
    day: int
    open_orders: np.ndarray  # global order ids open today (sorted)
    arrived: np.ndarray  # ids that arrived today
    cancelled: np.ndarray  # ids cancelled today
    pool_capacity: np.ndarray
    agent_capacity: np.ndarray
    events: list[str] = field(default_factory=list)


class World:
    def __init__(self, scn: Scenario, cfg: WorldConfig = WorldConfig()):
        self.scn, self.cfg = scn, cfg
        rng = np.random.default_rng(cfg.seed)
        n, D = scn.n_orders, cfg.days
        initial = rng.random(n) < cfg.initial_share
        self.arrival = np.where(initial, 0, rng.integers(1, max(D, 2), size=n))
        # cancellation day: geometric after arrival (inf if beyond horizon)
        gaps = rng.geometric(cfg.cancel_rate, size=n) if cfg.cancel_rate > 0 else np.full(n, 10**9)
        self.cancel = self.arrival + gaps
        walk = rng.normal(0, cfg.supply_walk, size=(D, scn.n_pools)).cumsum(axis=0)
        self.supply_factor = np.clip(1 + walk, 0.85, 1.10)
        self.shock_day = cfg.shock_day if cfg.shock_day is not None else D // 2

    def day(self, d: int) -> DayWorld:
        scn, cfg = self.scn, self.cfg
        open_mask = (self.arrival <= d) & (self.cancel > d)
        open_orders = np.flatnonzero(open_mask)
        arrived = np.flatnonzero(self.arrival == d) if d > 0 else np.flatnonzero(self.arrival == 0)
        cancelled = np.flatnonzero(self.cancel == d)
        share = (self.arrival <= d).mean()
        pool = np.floor(scn.pool_capacity * self.supply_factor[d] * share).astype(np.int64)
        agent = scn.agent_capacity.astype(float).copy()
        events = []
        if self.shock_day <= d < self.shock_day + cfg.shock_len:
            agent[cfg.shock_agent] *= 1 - cfg.shock_size
            events.append(
                f"{scn.agent_names[cfg.shock_agent]} capacity -{cfg.shock_size:.0%} (rail disruption)"
            )
        agent_cap = np.floor(agent * share).astype(np.int64)
        if d > 0:
            events.append(f"{len(arrived)} new orders, {len(cancelled)} cancellations")
        return DayWorld(d, open_orders, arrived, cancelled, pool, agent_cap, events)
