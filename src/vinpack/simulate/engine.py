"""Rolling daily re-optimisation: run ``solve_day`` for every day of the horizon.

Each day is an idempotent task keyed by (run_id, day) - the same shape an
Airflow DAG would have (one task per day, each depending on yesterday's plan),
kept in-process so the project runs with no scheduler to install.
"""

from __future__ import annotations

from collections.abc import Callable

from vinpack.pipeline.adapter import Scenario
from vinpack.pipeline.run import DayParams, DayPlan, solve_day
from vinpack.simulate.diff import plan_diff
from vinpack.simulate.scenarios import World, WorldConfig


def simulate(
    scn: Scenario,
    cfg: WorldConfig = WorldConfig(),
    params: DayParams = DayParams(),
    on_day: Callable[[DayPlan, list[dict]], None] | None = None,
) -> tuple[list[DayPlan], list[list[dict]]]:
    world = World(scn, cfg)
    plans: list[DayPlan] = []
    diffs: list[list[dict]] = []
    prev = None
    for d in range(cfg.days):
        w = world.day(d)
        plan = solve_day(scn, w, prev, params)
        diff = plan_diff(scn, w, prev, plan)
        plans.append(plan)
        diffs.append(diff)
        if on_day:
            on_day(plan, diff)
        prev = plan
    return plans, diffs


def run_and_store(con, run_id: str, scn: Scenario, cfg: WorldConfig, params: DayParams,
                  on_day: Callable[[DayPlan, list[dict]], None] | None = None):
    """Simulate the horizon and persist every day (idempotent per run_id/day)."""
    from vinpack import store

    store.start_run(con, run_id, cfg, params)

    def _save(plan: DayPlan, diff: list[dict]) -> None:
        store.save_day(con, run_id, scn, plan, diff)
        if on_day:
            on_day(plan, diff)

    return simulate(scn, cfg, params, _save)


def replay_from(con, base_run: str, new_run: str, day: int, params: DayParams,
                scn: Scenario) -> list[DayPlan]:
    """Apply an approved override from ``day`` on: copy earlier days, re-solve the rest."""
    from vinpack import store
    from vinpack.explain.evidence import Evidence

    ev = Evidence(con, base_run, scn)
    cfg = ev.cfg
    store.start_run(con, new_run, cfg, params)
    for t in ("kpis", "decisions", "pools", "diffs", "events"):
        cols = ", ".join(c for c in con.execute(f"SELECT * FROM {t} LIMIT 0").df().columns
                         if c != "run_id")
        con.execute(f"INSERT INTO {t} SELECT ? AS run_id, {cols} FROM {t} "
                    f"WHERE run_id = ? AND day < ?", [new_run, base_run, day])
    world = World(scn, cfg)
    prev = ev.plan(day - 1) if day > 0 else None
    plans = []
    for d in range(day, cfg.days):
        w = world.day(d)
        plan = solve_day(scn, w, prev, params)
        store.save_day(con, new_run, scn, plan, plan_diff(scn, w, prev, plan))
        plans.append(plan)
        prev = plan
    return plans
