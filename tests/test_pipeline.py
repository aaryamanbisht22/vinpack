"""End-to-end daily pipeline: stability behaviour, store round-trip, idempotency, replay."""

import numpy as np
import pytest

from vinpack import store
from vinpack.pipeline.run import DayParams, solve_day
from vinpack.simulate.engine import replay_from, run_and_store, simulate
from vinpack.simulate.scenarios import World, WorldConfig

CALM = dict(days=5, cancel_rate=0.0, supply_walk=0.0, shock_size=0.0)  # supply only grows


def test_huge_churn_weight_means_zero_churn(scenario):
    plans, _ = simulate(scenario, WorldConfig(**CALM), DayParams(churn_weight=50.0))
    assert sum(p.kpis["churn_l1"] for p in plans) == 0
    assert sum(p.kpis["broken_promises"] for p in plans) == 0


def test_zero_churn_weight_equals_fresh_solve(scenario):
    cfg = WorldConfig(**CALM)
    params = DayParams(churn_weight=0.0, lock_after=10**6, mip_gap=1e-9, l1_time=30)
    plans, _ = simulate(scenario, cfg, params)
    fresh = solve_day(scenario, World(scenario, cfg).day(3), None, params)
    assert fresh.kpis["value"] == plans[3].kpis["value"]


def test_plan_is_consistent(scenario):
    plans, diffs = simulate(scenario, WorldConfig(days=4), DayParams())
    for p in plans:
        filled = np.flatnonzero(p.filled)
        assert set(filled) <= set(p.open_orders)
        # supply respected
        assert np.all(scenario.pool_use[:, filled].sum(axis=1) <= p.pool_capacity)
        # every matched order is filled and loaded on a truck
        m = p.agent >= 0
        assert np.all(p.filled[m]) and np.all(p.truck[m] >= 0)
        assert p.kpis["trucks"] >= p.kpis["trucks_lb"]
    assert any(diffs[1:])


@pytest.fixture()
def db(tmp_path, scenario):
    con = store.connect(tmp_path / "t.duckdb")
    run_and_store(con, "t", scenario, WorldConfig(days=4), DayParams())
    yield con
    con.close()


def test_store_round_trip_and_idempotent(db, scenario):
    k1 = store.table(db, "kpis", "t")
    assert list(k1.day) == [0, 1, 2, 3]
    run_and_store(db, "t", scenario, WorldConfig(days=4), DayParams())  # re-run same id
    k2 = store.table(db, "kpis", "t")
    assert len(k2) == 4 and np.allclose(k1.value, k2.value)
    assert len(store.table(db, "pools", "t", 2)) == scenario.n_pools
    assert len(store.table(db, "decisions", "t", 2)) == scenario.n_orders


def test_replay_applies_override_from_day(db, scenario):
    from vinpack.explain.evidence import Evidence

    ev = Evidence(db, "t", scenario)
    p2 = ev.plan(2)
    target = int(next(j for j in p2.open_orders if not p2.filled[j]))
    params = DayParams(force_fill=(target,))
    replay_from(db, "t", "t+ovr", 2, params, scenario)
    new = Evidence(db, "t+ovr", scenario)
    assert np.array_equal(new.plan(1).filled, ev.plan(1).filled)  # earlier days copied
    assert new.plan(2).filled[target]
