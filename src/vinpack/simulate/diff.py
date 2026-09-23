"""What changed between yesterday's plan and today's, and which constraint caused it.

Every row carries a machine-readable ``cause`` plus a short human reason built
only from solver facts (LP shadow prices, reduced costs, capacity changes), so
the explainer can cite it without inventing anything.
"""

from __future__ import annotations

import numpy as np

from vinpack.pipeline.adapter import Scenario
from vinpack.pipeline.run import DayPlan
from vinpack.simulate.scenarios import DayWorld


def binding_pool(scn: Scenario, plan: DayPlan, order: int) -> tuple[int, float]:
    """The supply pool whose shadow price makes this order most expensive: argmax_i u_i r_ij."""
    charge = plan.l1_duals * scn.pool_use[:, order]
    i = int(np.argmax(charge))
    return i, float(charge[i])


def plan_diff(scn: Scenario, world: DayWorld, prev: DayPlan | None, plan: DayPlan) -> list[dict]:
    rows: list[dict] = []
    d = plan.day

    def row(order, change, cause, reason, frm=-1, to=-1):
        rows.append({"day": d, "order_id": int(order), "change": change, "cause": cause,
                     "from_agent": int(frm), "to_agent": int(to), "reason": reason})

    if prev is None:
        return rows
    for j in world.cancelled:
        if prev.filled[j]:
            row(j, "cancelled", "customer_cancel",
                "Customer cancelled; supply returned to the pool.", prev.agent[j], -1)
    both = np.intersect1d(plan.open_orders, prev.open_orders)
    for j in both:
        was, now = bool(prev.filled[j]), bool(plan.filled[j])
        if was and not now:
            i, charge = binding_pool(scn, plan, j)
            rc = plan.l1_reduced_costs[j]
            if rc < 0:
                reason = (f"Priced out: value {scn.values[j]} < shadow cost of its supply "
                          f"(reduced cost {rc:.0f}); tightest pool '{scn.pool_names[i]}' "
                          f"charges {charge:.0f}.")
                cause = f"pool:{i}"
            else:
                reason = (f"Displaced: still profitable in the LP (reduced cost {rc:+.0f}) but "
                          f"lost the integer trade-off for '{scn.pool_names[i]}' to other orders.")
                cause = f"displaced:{i}"
            row(j, "dropped", cause, reason, prev.agent[j], -1)
        elif now and not was:
            i, _ = binding_pool(scn, plan, j)
            row(j, "filled", "supply_freed",
                f"Filled: supply loosened (arrivals/cancellations); reduced cost "
                f"{plan.l1_reduced_costs[j]:+.0f}.", -1, plan.agent[j])
        elif now and was and prev.agent[j] >= 0 and plan.agent[j] != prev.agent[j]:
            a, b = int(prev.agent[j]), int(plan.agent[j])
            cap_cut = plan.agent_capacity[a] < prev.agent_capacity[a]
            if b < 0:
                cause, why = "held", "no DC capacity left today; held for tomorrow"
            elif cap_cut:
                cause = f"agent_capacity:{a}"
                why = (f"{scn.agent_names[a]} capacity cut "
                       f"{prev.agent_capacity[a]}->{plan.agent_capacity[a]}")
            else:
                delta = scn.match_cost[b, j] - scn.match_cost[a, j]
                cause = "rebalance"
                why = f"network rebalance (lane cost {delta:+d}) to fit today's order mix"
            row(j, "reassigned", cause, f"Moved {scn.agent_names[a]} -> "
                f"{scn.agent_names[b] if b >= 0 else 'HOLD'}: {why}.", a, b)
    for j in world.arrived:
        if plan.filled[j] and d > 0:
            row(j, "new_filled", "arrival", "New order, filled on arrival.", -1, plan.agent[j])
    return rows
