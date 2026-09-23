"""DuckDB persistence for simulation runs.

Writes are idempotent per (run_id, day): re-running a day replaces its rows,
so the daily task can be retried safely.

Tables
  runs       run_id, created_at, churn_weight, seed, days, backend, config (JSON)
  kpis       run_id, day, <kpi columns>
  decisions  run_id, day, order_id, is_open, filled, agent, truck, streak, reduced_cost, value
  pools      run_id, day, pool, name, capacity, used, shadow_price
  diffs      run_id, day, order_id, change, cause, from_agent, to_agent, reason
  events     run_id, day, event
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from vinpack.pipeline.adapter import Scenario
from vinpack.pipeline.run import DayParams, DayPlan
from vinpack.simulate.scenarios import WorldConfig

DEFAULT_DB = Path(__file__).resolve().parents[2] / "results" / "vinpack.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id VARCHAR PRIMARY KEY, created_at TIMESTAMP,
    churn_weight DOUBLE, seed INTEGER, days INTEGER, backend VARCHAR, config VARCHAR);
CREATE TABLE IF NOT EXISTS kpis (run_id VARCHAR, day INTEGER, open INTEGER, filled INTEGER,
    value DOUBLE, l1_lp_bound DOUBLE, l1_gap DOUBLE, l1_status VARCHAR, match_cost DOUBLE,
    l2_status VARCHAR, held INTEGER, trucks INTEGER, trucks_lb INTEGER, l3_proven INTEGER,
    churn_l1 INTEGER, churn_l2 INTEGER, broken_promises INTEGER, runtime DOUBLE);
CREATE TABLE IF NOT EXISTS decisions (run_id VARCHAR, day INTEGER, order_id INTEGER,
    is_open BOOLEAN, filled BOOLEAN, agent INTEGER, truck INTEGER, streak INTEGER,
    reduced_cost DOUBLE, value DOUBLE);
CREATE TABLE IF NOT EXISTS pools (run_id VARCHAR, day INTEGER, pool INTEGER, name VARCHAR,
    capacity DOUBLE, used DOUBLE, shadow_price DOUBLE);
CREATE TABLE IF NOT EXISTS diffs (run_id VARCHAR, day INTEGER, order_id INTEGER,
    change VARCHAR, cause VARCHAR, from_agent INTEGER, to_agent INTEGER, reason VARCHAR);
CREATE TABLE IF NOT EXISTS events (run_id VARCHAR, day INTEGER, event VARCHAR);
"""
_KPI_COLS = ["open", "filled", "value", "l1_lp_bound", "l1_gap", "l1_status", "match_cost",
             "l2_status", "held", "trucks", "trucks_lb", "l3_proven", "churn_l1", "churn_l2",
             "broken_promises", "runtime"]


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(_SCHEMA)
    return con


def start_run(con, run_id: str, cfg: WorldConfig, params: DayParams) -> None:
    con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
    for t in ("kpis", "decisions", "pools", "diffs", "events"):
        con.execute(f"DELETE FROM {t} WHERE run_id = ?", [run_id])
    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        [run_id, datetime.now(UTC).replace(tzinfo=None), params.churn_weight, cfg.seed,
         cfg.days, params.backend,
         json.dumps({"world": asdict(cfg), "params": asdict(params)})],
    )


def save_day(con, run_id: str, scn: Scenario, plan: DayPlan, diff: list[dict]) -> None:
    d = plan.day
    for t in ("kpis", "decisions", "pools", "diffs", "events"):
        con.execute(f"DELETE FROM {t} WHERE run_id = ? AND day = ?", [run_id, d])
    k = plan.kpis
    con.execute(f"INSERT INTO kpis VALUES ({', '.join(['?'] * (len(_KPI_COLS) + 2))})",
                [run_id, d] + [k[c] for c in _KPI_COLS])
    N = scn.n_orders
    is_open = np.zeros(N, dtype=bool)
    is_open[plan.open_orders] = True
    dec = pd.DataFrame({
        "run_id": run_id, "day": d, "order_id": np.arange(N), "is_open": is_open,
        "filled": plan.filled, "agent": plan.agent, "truck": plan.truck, "streak": plan.streak,
        "reduced_cost": plan.l1_reduced_costs, "value": scn.values.astype(float),
    })
    _insert(con, "decisions", dec)
    used = scn.pool_use[:, plan.filled].sum(axis=1)
    pool_df = pd.DataFrame({
        "run_id": run_id, "day": d, "pool": np.arange(scn.n_pools), "name": list(scn.pool_names),
        "capacity": plan.pool_capacity.astype(float), "used": used.astype(float),
        "shadow_price": plan.l1_duals,
    })
    _insert(con, "pools", pool_df)
    if diff:
        df = pd.DataFrame(diff)
        df.insert(0, "run_id", run_id)
        df = df[["run_id", "day", "order_id", "change", "cause", "from_agent", "to_agent",
                 "reason"]]
        _insert(con, "diffs", df)
    for e in plan.events:
        con.execute("INSERT INTO events VALUES (?, ?, ?)", [run_id, d, e])


def _insert(con, table_name: str, df: pd.DataFrame) -> None:
    """Insert a DataFrame via an explicitly registered view (no name-based replacement scans)."""
    con.register("_vinpack_insert", df)
    try:
        con.execute(f"INSERT INTO {table_name} SELECT * FROM _vinpack_insert")
    finally:
        con.unregister("_vinpack_insert")


def runs(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM runs ORDER BY created_at DESC").df()


def table(con, name: str, run_id: str, day: int | None = None) -> pd.DataFrame:
    q = f"SELECT * FROM {name} WHERE run_id = ?"
    args: list = [run_id]
    if day is not None:
        q += " AND day = ?"
        args.append(day)
    return con.execute(q + " ORDER BY day", args).df()


def run_config(con, run_id: str) -> tuple[WorldConfig, DayParams]:
    row = con.execute("SELECT config FROM runs WHERE run_id = ?", [run_id]).fetchone()
    if row is None:
        raise KeyError(f"unknown run {run_id!r}")
    cfg = json.loads(row[0])
    params = cfg["params"]
    params["force_fill"] = tuple(params.get("force_fill", ()))
    params["extra_supply"] = tuple(tuple(x) for x in params.get("extra_supply", ()))
    return WorldConfig(**cfg["world"]), DayParams(**params)
