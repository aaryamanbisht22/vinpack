"""Deterministic explanations used when no Claude credentials are configured.

They read the same evidence the agent sees, so the answer is always grounded;
they just cannot follow up or propose a trade-off on their own.
"""

from __future__ import annotations


def explain_order(ev, day: int, order_id: int) -> str:
    o = ev.get_order(day, order_id)
    if "error" in o:
        return o["error"]
    lines = [f"**Order {order_id} on day {day}** - value {o['value']}, arrived day {o['arrival_day']}."]
    if not o["is_open"]:
        lines.append("It is not open today (not yet arrived, or cancelled).")
        return "\n\n".join(lines)
    if o["filled"]:
        lines.append(
            f"It is **filled**, matched to **{o['agent']}**, truck #{o['truck']}, "
            f"filled {o['days_filled_in_a_row']} day(s) in a row"
            + (" (promised to the customer)." if o["promised"] else ".")
        )
    else:
        top = o["supply_charges"][0]
        rc = o["reduced_cost"]
        lines.append(
            f"It is **not filled**. Its LP reduced cost is {rc:+.2f}: the shadow prices of the "
            f"supply it would consume total {o['total_supply_charge']:.2f} against a value of "
            f"{o['value']}. The most expensive pool is **{top['pool']}** ({top['units_used']} units "
            f"x shadow price {top['shadow_price']:.4f} = {top['charge']:.2f})."
        )
        if rc is not None and rc >= 0:
            lines.append("It is profitable in the LP relaxation, so it lost the integer trade-off "
                         "(or the churn penalty kept yesterday's plan).")
        wf = ev.what_if_force(day, order_id)
        if "value_delta" in wf:
            lines.append(
                f"What-if (proposal {wf.get('proposal_id')}): forcing it in changes plan value by "
                f"{wf['value_delta']:+d}, displaces {[d['order_id'] for d in wf['orders_displaced']]}"
                f", newly fills {wf['orders_newly_filled']}, and moves churn "
                f"{wf['churn_before']} -> {wf['churn_after']} (penalty "
                f"{wf['churn_penalty_per_change']:.0f} each). Net objective change "
                f"{wf['objective_delta_after_churn_penalty']:+.0f}: {wf['verdict']}."
            )
    if o["history"]:
        lines.append("History: " + "; ".join(f"day {h['day']} {h['change']}" for h in o["history"]))
    return "\n\n".join(lines)
