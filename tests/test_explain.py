"""Explainer: evidence is solver-consistent, and the agent loop only cites tool facts."""

from types import SimpleNamespace

import pytest

from vinpack import store
from vinpack.explain import agent, templates
from vinpack.explain.evidence import Evidence
from vinpack.pipeline.run import DayParams
from vinpack.simulate.engine import run_and_store
from vinpack.simulate.scenarios import WorldConfig


@pytest.fixture(scope="module")
def ev(tmp_path_factory, scenario):
    con = store.connect(tmp_path_factory.mktemp("db") / "e.duckdb")
    run_and_store(con, "e", scenario, WorldConfig(days=5), DayParams())
    return Evidence(con, "e", scenario)


def _unfilled(ev, day):
    p = ev.plan(day)
    return int(next(j for j in p.open_orders if not p.filled[j]))


def test_reduced_cost_equals_value_minus_supply_charge(ev, scenario):
    j = _unfilled(ev, 4)
    o = ev.get_order(4, j)
    assert abs(o["reduced_cost"] - (o["value"] - o["total_supply_charge"])) < 0.02


def test_what_if_reports_churn_adjusted_objective(ev):
    j = _unfilled(ev, 4)
    wf = ev.what_if_force(4, j)
    assert j in wf["orders_newly_filled"]
    lam = wf["churn_penalty_per_change"]
    expect = wf["value_delta"] - lam * (wf["churn_after"] - wf["churn_before"])
    assert abs(wf["objective_delta_after_churn_penalty"] - expect) < 0.05  # lambda is rounded
    # the stored plan was optimal for the churn-penalised objective, so forcing cannot beat it
    assert wf["objective_delta_after_churn_penalty"] <= 1e-6 * abs(wf["value_before"]) + 1
    assert wf["proposal_id"] in ev.proposals


def test_template_explainer_runs(ev):
    text = templates.explain_order(ev, 4, _unfilled(ev, 4))
    assert "not filled" in text and "reduced cost" in text


def test_verify_numbers_flags_invented_values():
    sources = ['{"value": 575, "reduced_cost": 1.11, "shadow_price": 0.262}']
    assert agent.verify_numbers("Value 575, reduced cost +1.11.", sources) == []
    assert agent.verify_numbers("Value 575 but it would earn 9,999 more.", sources) == ["9,999"]


class FakeClient:
    """Replays a scripted conversation: one tool call, then a final answer."""

    def __init__(self, order, day):
        self.calls = []
        self.script = [
            SimpleNamespace(stop_reason="tool_use", content=[
                SimpleNamespace(type="tool_use", id="t1", name="get_order",
                                input={"day": day, "order_id": order})]),
            None,  # final answer is built from the tool output below
        ]
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script[len(self.calls) - 1]
        if step is not None:
            return step
        payload = kwargs["messages"][-1]["content"][0]["content"]
        import json

        o = json.loads(payload)
        text = (f"Not filled: value {o['value']} vs supply charge {o['total_supply_charge']}; "
                f"it would add 123456 value.")
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def test_agent_loop_dispatches_tools_and_flags_unsupported_numbers(ev):
    j = _unfilled(ev, 4)
    client = FakeClient(j, 4)
    ans = agent.ask(ev, f"Why is order {j} not filled on day 4?", client=client)
    assert ans.tool_calls[0]["tool"] == "get_order"
    assert ans.unverified == ["123456"]  # invented number caught, real ones pass
    first = client.calls[0]
    assert first["model"] == agent.MODEL and first["tools"] == agent.TOOLS
    assert all(t["strict"] for t in agent.TOOLS)
