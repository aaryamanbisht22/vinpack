"""Solver-backed facts about a stored run: the only thing the explainer may cite.

Every method returns plain JSON-able dicts built from the stored plan, LP
shadow prices, or a fresh what-if re-solve of the same day. What-ifs rebuild the
day exactly: the world is a pure function of (seed, day), and yesterday's plan
is read back from the store.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vinpack import store
from vinpack.pipeline.adapter import Scenario, load_default
from vinpack.pipeline.run import DayParams, DayPlan, solve_day
from vinpack.simulate.scenarios import World


class Evidence:
    def __init__(self, con, run_id: str, scn: Scenario | None = None):
        self.con, self.run_id = con, run_id
        self.scn = scn or load_default()
        self.cfg, self.params = store.run_config(con, run_id)
        self.world = World(self.scn, self.cfg)
        self._plans: dict[int, DayPlan] = {}
        self.proposals: dict[str, dict] = {}

    # ------------------------------------------------------------------ plan loading
    def plan(self, day: int) -> DayPlan:
        if day in self._plans:
            return self._plans[day]
        dec = store.table(self.con, "decisions", self.run_id, day)
        if dec.empty:
            raise KeyError(f"run {self.run_id!r} has no day {day}")
        pools = store.table(self.con, "pools", self.run_id, day)
        kp = store.table(self.con, "kpis", self.run_id, day).iloc[0].to_dict()
        w = self.world.day(day)
        plan = DayPlan(
            day=day,
            open_orders=dec.loc[dec.is_open, "order_id"].to_numpy(),
            filled=dec["filled"].to_numpy(bool),
            agent=dec["agent"].to_numpy(int),
            truck=dec["truck"].to_numpy(int),
            streak=dec["streak"].to_numpy(int),
            kpis=kp,
            pool_capacity=pools["capacity"].to_numpy(),
            agent_capacity=w.agent_capacity,
            events=list(w.events),
            l1_duals=pools["shadow_price"].to_numpy(),
            l1_reduced_costs=dec["reduced_cost"].to_numpy(float),
        )
        self._plans[day] = plan
        return plan

    def days(self) -> list[int]:
        return [int(d) for d in store.table(self.con, "kpis", self.run_id)["day"]]

    # ------------------------------------------------------------------ tools
    def get_order(self, day: int, order_id: int) -> dict:
        scn, plan = self.scn, self.plan(day)
        j = int(order_id)
        if not 0 <= j < scn.n_orders:
            return {"error": f"order {j} does not exist (0..{scn.n_orders - 1})"}
        is_open = bool(np.isin(j, plan.open_orders))
        charges = plan.l1_duals * scn.pool_use[:, j]
        top = np.argsort(-charges)[:3]
        out = {
            "day": day,
            "order_id": j,
            "value": int(scn.values[j]),
            "arrival_day": int(self.world.arrival[j]),
            "cancel_day": int(self.world.cancel[j]) if self.world.cancel[j] < self.cfg.days else None,
            "is_open": is_open,
            "filled": bool(plan.filled[j]),
            "agent": scn.agent_names[plan.agent[j]] if plan.agent[j] >= 0 else None,
            "truck": int(plan.truck[j]) if plan.truck[j] >= 0 else None,
            "days_filled_in_a_row": int(plan.streak[j]),
            "promised": bool(plan.streak[j] >= self.params.lock_after),
            "reduced_cost": None if np.isnan(plan.l1_reduced_costs[j])
            else round(float(plan.l1_reduced_costs[j]), 2),
            "supply_charges": [
                {"pool": scn.pool_names[i], "units_used": int(scn.pool_use[i, j]),
                 "shadow_price": round(float(plan.l1_duals[i]), 4),
                 "charge": round(float(charges[i]), 2)}
                for i in top
            ],
            "total_supply_charge": round(float(charges.sum()), 2),
        }
        if plan.agent[j] >= 0:
            costs = scn.match_cost[:, j]
            out["lane_costs"] = {scn.agent_names[i]: int(costs[i]) for i in np.argsort(costs)[:4]}
        out["history"] = [
            {"day": int(r.day), "change": r.change, "reason": r.reason}
            for r in store.table(self.con, "diffs", self.run_id).query("order_id == @j").itertuples()
        ]
        return out

    def binding_constraints(self, day: int) -> dict:
        scn, plan = self.scn, self.plan(day)
        pools = store.table(self.con, "pools", self.run_id, day)
        pool_rows = [
            {"pool": r.name, "capacity": int(r.capacity), "used": int(r.used),
             "slack": int(r.capacity - r.used), "shadow_price": round(float(r.shadow_price), 4)}
            for r in pools.sort_values("shadow_price", ascending=False).itertuples()
        ]
        load = np.zeros(scn.n_agents)
        m = plan.agent >= 0
        np.add.at(load, plan.agent[m], scn.match_load[plan.agent[m], np.flatnonzero(m)])
        dc_rows = [
            {"dc": scn.agent_names[i], "capacity": int(plan.agent_capacity[i]),
             "used": int(load[i]), "utilisation": round(float(load[i] / max(1, plan.agent_capacity[i])), 3)}
            for i in np.argsort(-load / np.maximum(plan.agent_capacity, 1))
        ]
        return {"day": day, "supply_pools": pool_rows, "delivery_centers": dc_rows,
                "events": plan.events}

    def plan_diff(self, day: int) -> dict:
        diffs = store.table(self.con, "diffs", self.run_id, day)
        return {
            "day": day,
            "counts": diffs["change"].value_counts().to_dict() if not diffs.empty else {},
            "changes": diffs[["order_id", "change", "reason"]].head(40).to_dict("records"),
        }

    def kpis(self, day: int) -> dict:
        k = store.table(self.con, "kpis", self.run_id, day).iloc[0].to_dict()
        return {key: (round(v, 4) if isinstance(v, float) else v) for key, v in k.items()
                if key != "run_id"}

    def _resolve(self, day: int, params: DayParams) -> DayPlan:
        prev = self.plan(day - 1) if day > 0 else None
        return solve_day(self.scn, self.world.day(day), prev, params)

    def what_if_force(self, day: int, order_id: int) -> dict:
        """Re-solve ``day`` with the order forced into the plan (planner override)."""
        j = int(order_id)
        base = self.plan(day)
        if not np.isin(j, base.open_orders):
            return {"error": f"order {j} is not open on day {day}"}
        if base.filled[j]:
            return {"note": f"order {j} is already filled on day {day}", "value_delta": 0}
        params = replace(self.params, force_fill=tuple(self.params.force_fill) + (j,))
        try:
            alt = self._resolve(day, params)
        except ValueError as e:
            return {"error": str(e)}
        return self._compare(day, base, alt, {"type": "force_fill", "order_id": j}, params)

    def what_if_capacity(self, day: int, pool: str | int, units: int, order_id: int | None = None) -> dict:
        """Re-solve ``day`` with ``units`` more of one supply pool."""
        i = self._pool_index(pool)
        if i is None:
            return {"error": f"unknown pool {pool!r}; pools are {list(self.scn.pool_names)}"}
        base = self.plan(day)
        params = replace(self.params,
                         extra_supply=tuple(self.params.extra_supply) + ((i, int(units)),))
        alt = self._resolve(day, params)
        out = self._compare(day, base, alt,
                            {"type": "extra_supply", "pool": self.scn.pool_names[i],
                             "units": int(units)}, params)
        if order_id is not None:
            out["order_filled_after"] = bool(alt.filled[int(order_id)])
        return out

    def _pool_index(self, pool) -> int | None:
        if isinstance(pool, int) or (isinstance(pool, str) and pool.isdigit()):
            i = int(pool)
            return i if 0 <= i < self.scn.n_pools else None
        for i, name in enumerate(self.scn.pool_names):
            if pool.lower() in name.lower():
                return i
        return None

    def _compare(self, day, base: DayPlan, alt: DayPlan, action: dict, params: DayParams) -> dict:
        scn = self.scn
        newly = np.flatnonzero(alt.filled & ~base.filled)
        displaced = np.flatnonzero(base.filled & ~alt.filled)
        pid = f"p{len(self.proposals) + 1}"
        self.proposals[pid] = {"day": day, "action": action, "params": params}
        # The daily objective is value - lambda * churn; report both so a what-if that
        # buys value by reshuffling yesterday's plan is not mistaken for a free win.
        lam = self.params.churn_weight * float(scn.values.mean())
        churn_before, churn_after = int(base.kpis["churn_l1"]), int(alt.kpis["churn_l1"])
        value_delta = int(alt.kpis["value"] - base.kpis["value"])
        objective_delta = value_delta - lam * (churn_after - churn_before)
        return {
            "proposal_id": pid,
            "action": action,
            "value_before": int(base.kpis["value"]),
            "value_after": int(alt.kpis["value"]),
            "value_delta": value_delta,
            "churn_before": churn_before,
            "churn_after": churn_after,
            "churn_penalty_per_change": round(lam, 2),
            "objective_delta_after_churn_penalty": round(objective_delta, 2),
            "verdict": ("improves today's objective" if objective_delta > 0 else
                        "worse than the current plan once the churn penalty is counted"),
            "orders_newly_filled": [int(x) for x in newly],
            "orders_displaced": [{"order_id": int(x), "value": int(scn.values[x])} for x in displaced],
            "trucks_before": int(base.kpis["trucks"]),
            "trucks_after": int(alt.kpis["trucks"]),
        }
