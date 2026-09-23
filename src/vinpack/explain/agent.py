"""Claude agent that answers planner questions using only solver-backed evidence.

The agent can read (get_order, binding_constraints, plan_diff, kpis) and
*propose* (what_if_force, what_if_capacity). A proposal is a fully re-solved
alternative plan; nothing changes until a human approves it in the UI, which
then replays the run from that day with the override (``replay_from``).

Correctness guard: after the answer is written, every number in it is checked
against the numbers that appeared in tool outputs (and the question). Numbers
with no source are returned as ``unverified`` so the UI can flag them - the
"correct, not merely plausible" rule from the brief this project answers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from vinpack.explain.evidence import Evidence

MODEL = os.environ.get("VINPACK_MODEL", "claude-opus-5")

SYSTEM = """You are the planning analyst for a vehicle demand-planning optimiser.
Each day the optimiser (1) allocates scarce supply to customer orders (multidimensional
knapsack), (2) matches filled orders to delivery centers (generalized assignment), and
(3) loads them onto carriers (bin packing), with a penalty for changing yesterday's plan.

Answer the planner's question using ONLY facts returned by your tools. Every number you
state must come from a tool result. Explain with the optimiser's own logic: LP shadow
prices and reduced costs for allocation, DC capacity and lane costs for matching.
If a change would help, run a what-if and present it as a proposal (quote its
proposal_id, value delta, and displaced orders); never claim it has been applied - a
human must approve it. Keep answers short: a verdict line, then 2-5 bullets."""

TOOLS = [
    {
        "name": "get_order",
        "description": "Status of one order on one day: filled?, DC, truck, promise status, value, "
                       "LP reduced cost, top supply-pool charges (shadow price x units), lane "
                       "costs, and its change history.",
        "input_schema": {
            "type": "object",
            "properties": {"day": {"type": "integer"}, "order_id": {"type": "integer"}},
            "required": ["day", "order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "binding_constraints",
        "description": "Supply pools (capacity, used, slack, shadow price) and delivery-center "
                       "utilisation for a day, plus that day's events (e.g. capacity shocks).",
        "input_schema": {
            "type": "object",
            "properties": {"day": {"type": "integer"}},
            "required": ["day"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "plan_diff",
        "description": "What changed versus the previous day (filled, dropped, reassigned, "
                       "cancelled) with the solver's reason for each change.",
        "input_schema": {
            "type": "object",
            "properties": {"day": {"type": "integer"}},
            "required": ["day"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "kpis",
        "description": "Day-level KPIs: open/filled orders, plan value, LP bound, MIP gap, "
                       "matching cost, trucks vs lower bound, churn, broken promises.",
        "input_schema": {
            "type": "object",
            "properties": {"day": {"type": "integer"}},
            "required": ["day"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "what_if_force",
        "description": "Re-solve the day with an unfilled order forced in. Returns a proposal "
                       "(value delta, displaced orders, trucks). Does NOT apply it.",
        "input_schema": {
            "type": "object",
            "properties": {"day": {"type": "integer"}, "order_id": {"type": "integer"}},
            "required": ["day", "order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "what_if_capacity",
        "description": "Re-solve the day with extra units of one supply pool (name or index). "
                       "Optionally report whether a given order becomes filled. Returns a "
                       "proposal; does NOT apply it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "integer"},
                "pool": {"type": "string"},
                "units": {"type": "integer"},
                "order_id": {"type": ["integer", "null"]},
            },
            "required": ["day", "pool", "units", "order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


@dataclass
class AgentAnswer:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    model: str = MODEL


def run_tool(ev: Evidence, name: str, args: dict) -> dict:
    fn = {
        "get_order": ev.get_order,
        "binding_constraints": ev.binding_constraints,
        "plan_diff": ev.plan_diff,
        "kpis": ev.kpis,
        "what_if_force": ev.what_if_force,
        "what_if_capacity": ev.what_if_capacity,
    }.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except (KeyError, ValueError, TypeError) as e:
        return {"error": str(e)}


_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def verify_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in ``answer`` that appear in no source (small integers <= 10 are exempt:
    list counts, bullet numbers, 'top 3')."""
    known: set[float] = set()
    for s in sources:
        known |= numbers_in(s)
    bad = []
    for m in _NUM.findall(answer):
        v = round(float(m.replace(",", "")), 2)
        if abs(v) <= 10 or v in known or round(v) in known or abs(v) in known:
            continue
        # allow percentages and rounded forms of known values
        if any(abs(v - k) <= max(0.51, 0.005 * abs(k)) for k in known):
            continue
        bad.append(m)
    return bad


def ask(ev: Evidence, question: str, client=None, max_turns: int = 8) -> AgentAnswer:
    import anthropic

    client = client or anthropic.Anthropic()
    days = ev.days()
    context = (f"Run '{ev.run_id}', days {days[0]}..{days[-1]}, "
               f"{ev.scn.n_orders} orders, pools: {', '.join(ev.scn.pool_names)}.")
    messages: list[dict] = [{"role": "user", "content": f"{context}\n\nQuestion: {question}"}]
    calls: list[dict] = []
    sources = [context, question]
    response = None
    for _ in range(max_turns):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "adaptive"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return AgentAnswer("The model declined this request.", calls)
        if response.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            out = run_tool(ev, block.name, dict(block.input))
            payload = json.dumps(out, default=str)
            sources.append(payload)
            calls.append({"tool": block.name, "input": dict(block.input), "output": out})
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": payload,
                            **({"is_error": True} if "error" in out else {})})
        messages.append({"role": "user", "content": results})
    text = "\n".join(b.text for b in (response.content if response else []) if b.type == "text")
    proposals = [c["output"]["proposal_id"] for c in calls
                 if isinstance(c["output"], dict) and "proposal_id" in c["output"]]
    return AgentAnswer(text, calls, proposals, verify_numbers(text, sources))


def has_credentials() -> bool:
    """True if the SDK can find credentials: an env var or an `ant auth login` profile.

    (The client constructor does not raise without credentials, so check sources directly.)
    """
    from pathlib import Path

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()
