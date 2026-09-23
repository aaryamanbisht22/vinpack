"""vinpack command line.

    vinpack fetch                              download benchmark data
    vinpack simulate --days 14 --churn 0.5     rolling daily re-solve, stored in DuckDB
    vinpack pareto --churn 0,0.1,0.25,0.5,1,2  value-vs-churn frontier
    vinpack bench [--quick]                    per-layer benchmarks vs published optima
    vinpack explain --run default --day 9 --order 42
    vinpack app                                Streamlit dashboard
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()
ROOT = Path(__file__).resolve().parents[2]


@app.command()
def fetch(no_pisinger: bool = typer.Option(False, help="skip the 150 MB Pisinger archive")):
    """Download OR-Library and Pisinger instances into data/raw/."""
    from vinpack.io.fetch import main as fetch_main

    sys.argv = ["fetch"] + (["--no-pisinger"] if no_pisinger else [])
    fetch_main()


def _params(churn: float, backend: str, l1_time: float, l2_time: float):
    from vinpack.pipeline.run import DayParams

    return DayParams(churn_weight=churn, backend=backend, l1_time=l1_time, l2_time=l2_time)


@app.command()
def simulate(
    run_id: str = typer.Option("default", "--run", help="run id in the store"),
    days: int = 14,
    churn: float = typer.Option(0.5, help="churn penalty, x mean order value / lane cost"),
    seed: int = 7,
    backend: str = typer.Option("highs", help="MILP solver (highs)"),
    l1_time: float = 5.0,
    l2_time: float = 5.0,
    db: Path = typer.Option(None, help="DuckDB file (default results/vinpack.duckdb)"),
):
    """Run the rolling daily re-optimisation and store every day."""
    from vinpack import store
    from vinpack.pipeline.adapter import load_default
    from vinpack.simulate.engine import run_and_store
    from vinpack.simulate.scenarios import WorldConfig

    scn = load_default()
    con = store.connect(db)
    t = Table("day", "open", "filled", "value", "L1 gap", "trucks (LB)", "churn L1/L2",
              "broken", "sec", title=f"run '{run_id}'  (churn weight {churn})")

    def show(plan, diff):
        k = plan.kpis
        t.add_row(str(k["day"]), str(k["open"]), str(k["filled"]), f"{k['value']:,.0f}",
                  f"{k['l1_gap']:.2%}", f"{k['trucks']} ({k['trucks_lb']})",
                  f"{k['churn_l1']}/{k['churn_l2']}", str(k["broken_promises"]),
                  f"{k['runtime']:.1f}")
        console.print(f"  day {k['day']:>2}: {k['filled']}/{k['open']} filled, "
                      f"{len(diff)} changes, {k['runtime']:.1f}s", highlight=False)

    run_and_store(con, run_id, scn, WorldConfig(days=days, seed=seed),
                  _params(churn, backend, l1_time, l2_time), show)
    console.print(t)
    con.close()


@app.command()
def pareto(
    churn: str = typer.Option("0,0.1,0.25,0.5,1,2", help="comma-separated churn weights"),
    days: int = 14,
    seed: int = 7,
    backend: str = "highs",
    l1_time: float = 3.0,
    l2_time: float = 3.0,
    db: Path = None,
):
    """Sweep the churn weight; store each run as pareto-<w> and print the frontier."""
    from vinpack import store
    from vinpack.pipeline.adapter import load_default
    from vinpack.simulate.engine import run_and_store
    from vinpack.simulate.scenarios import WorldConfig

    scn = load_default()
    con = store.connect(db)
    t = Table("churn weight", "total value", "orders changed (L1)", "DC moves (L2)",
              "broken promises", "trucks", title="value vs. plan stability")
    for w in [float(x) for x in churn.split(",")]:
        console.print(f"churn weight {w} ...")
        plans, _ = run_and_store(con, f"pareto-{w:g}", scn, WorldConfig(days=days, seed=seed),
                                 _params(w, backend, l1_time, l2_time))
        tot = {k: sum(p.kpis[k] for p in plans)
               for k in ("value", "churn_l1", "churn_l2", "broken_promises", "trucks")}
        t.add_row(f"{w:g}", f"{tot['value']:,.0f}", str(tot["churn_l1"]), str(tot["churn_l2"]),
                  str(tot["broken_promises"]), str(tot["trucks"]))
    console.print(t)
    con.close()


@app.command()
def bench(
    quick: bool = typer.Option(False, help="small subset (CI-sized)"),
    backend: str = "highs",
    out: Path = typer.Option(ROOT / "results", help="output directory"),
):
    """Run every layer's algorithms on the unmodified benchmark instances."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.run_all import run

    run(quick=quick, backend=backend, out=out)


@app.command()
def explain(
    order: int = typer.Option(..., help="order id"),
    day: int = typer.Option(..., help="day"),
    run_id: str = typer.Option("default", "--run"),
    question: str = typer.Option(None, help="free-form question (default: why is/isn't it filled)"),
    db: Path = None,
):
    """Explain one order's status. Uses Claude if credentials exist, else a template."""
    from vinpack import store
    from vinpack.explain import agent, templates
    from vinpack.explain.evidence import Evidence

    con = store.connect(db)
    ev = Evidence(con, run_id)
    if agent.has_credentials():
        q = question or f"Why is order {order} filled or not filled on day {day}, and what would it take?"
        ans = agent.ask(ev, q)
        console.print(ans.text)
        console.print(f"\n[dim]{len(ans.tool_calls)} tool calls, model {ans.model}; "
                      f"proposals {ans.proposals or 'none'}[/dim]")
        if ans.unverified:
            console.print(f"[yellow]unverified numbers: {ans.unverified}[/yellow]")
    else:
        console.print("[dim](no Anthropic credentials found - deterministic explanation)[/dim]\n")
        console.print(templates.explain_order(ev, day, order))
    con.close()


@app.command(name="app")
def run_app(port: int = 8501):
    """Launch the Streamlit dashboard."""
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(ROOT / "app" / "streamlit_app.py"), "--server.port", str(port)],
                   check=False)


if __name__ == "__main__":
    app()
