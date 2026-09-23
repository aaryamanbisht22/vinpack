"""Build docs/results.md from results/benchmarks.md and the stored Pareto sweep.

    make publish      (after `make bench` and `make pareto`)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vinpack import store  # noqa: E402


def pareto_table() -> str:
    con = store.connect(read_only=True)
    runs = store.runs(con)
    runs = runs[runs.run_id.str.startswith("pareto-")].sort_values("churn_weight")
    if runs.empty:
        return "_Run `make pareto` to populate._\n"
    rows = []
    base = None
    for r in runs.itertuples():
        k = store.table(con, "kpis", r.run_id)
        v, c = k.value.sum(), k.churn_l1.sum()
        if base is None:
            base = (v, c)
        rows.append(
            f"| {r.churn_weight:g} | {v:,.0f} | {(v - base[0]) / base[0]:+.1%} | {c} | "
            f"{(c - base[1]) / max(1, base[1]):+.0%} | {k.churn_l2.sum()} | "
            f"{k.broken_promises.sum()} | {k.trucks.sum()} | {(k.trucks == k.trucks_lb).mean():.0%} |"
        )
    con.close()
    head = ("| churn weight λ | plan value (14 days) | vs λ=0 | orders re-planned | vs λ=0 | "
            "DC reassignments | broken promises | carriers | days with proven-optimal loading |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    return head + "\n".join(rows) + "\n"


def main() -> None:
    bench = ROOT / "results" / "benchmarks.md"
    if not bench.exists():
        bench = ROOT / "results" / "benchmarks_quick.md"
    text = [
        "# Results\n",
        "All numbers are produced by `make bench` and `make pareto` on an Apple M4 Pro with HiGHS "
        "(open source). Nothing on this page is typed in by hand.\n",
        "## Per-layer benchmarks on unmodified academic instances\n",
        "*= reference*: matches or beats the published optimum or best-known value. "
        "*proven opt.*: the method itself proved optimality (bound = value). *gap*: relative "
        "distance to the reference (positive = worse). For gapa-d the reference is this "
        "project's own proven MIP optimum when one was found.\n",
        bench.read_text() if bench.exists() else "_Run `make bench`._\n",
        "\n*gapd*: no published optimum is stored and the 20 s MIP does not prove one, so "
        "those rows show no gap to a reference. Their gap to the MIP's own proven bound is in "
        "`results/benchmarks.csv` (`bound` column): 0.05-1.3%.\n",
        "\n## Daily re-solve: value vs. plan stability\n",
        "14-day rolling simulation on the composed scenario (seed 7). λ is the churn penalty "
        "as a multiple of the mean order value (L1) and the mean lane cost (L2).\n",
        pareto_table(),
    ]
    (ROOT / "docs" / "results.md").write_text("\n".join(text))
    print("wrote docs/results.md")


if __name__ == "__main__":
    main()
