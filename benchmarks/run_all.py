"""Per-layer benchmarks on the unmodified academic instances.

    uv run vinpack bench            # full suite (~30-40 min on an M-series laptop)
    uv run vinpack bench --quick    # CI-sized subset (~1-2 min)

Writes results/benchmarks.csv (one row per instance x method) and
results/benchmarks.md (the summary table used in the README and docs).

Reference values:
  L0  Pisinger published optimum (and optimal solution vector)
  L1  Chu & Beasley best-known (mkcbres.txt); LP bound from the same file
  L2  gap1-12: published optimum. gapa-d: no stored reference - the reference is
      this project's own proven MIP optimum when the solver proves one (marked so).
  L3  Falkenauer best-known / proven optimum (binpack files)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vinpack.io.orlib import read_binpack, read_gap, read_mknap  # noqa: E402
from vinpack.io.pisinger import read_pisinger  # noqa: E402
from vinpack.solvers import bpp, gap, mkp  # noqa: E402
from vinpack.solvers.colgen import column_generation  # noqa: E402
from vinpack.solvers.kp_dp import kp_bb, solve_knapsack  # noqa: E402
from vinpack.solvers.lagrangian import gap_lagrangian, mkp_lagrangian  # noqa: E402

RAW = ROOT / "data" / "raw"


def _row(layer, family, inst, method, value, reference, ref_kind, sense, seconds,
         bound=np.nan, proven=False, feasible=True, **extra):
    if reference is None or not np.isfinite(value):
        gap_ref, match = np.nan, False
    elif sense == "max":
        gap_ref = (reference - value) / max(1, abs(reference))
        match = value >= reference - 1e-6
    else:
        gap_ref = (value - reference) / max(1, abs(reference))
        match = value <= reference + 1e-6
    return {"layer": layer, "family": family, "instance": inst, "method": method,
            "value": value, "reference": reference, "reference_kind": ref_kind,
            "matches_reference": bool(match), "gap_to_reference": gap_ref, "bound": bound,
            "proven_optimal": bool(proven), "feasible": bool(feasible), "seconds": seconds,
            **extra}


# ------------------------------------------------------------------------------ L0
def bench_l0(quick: bool) -> list[dict]:
    rows = []
    sizes = (50, 100, 200) if quick else (50, 100, 200, 500, 1000, 2000)
    per_file = 5 if quick else 20
    for t in (1, 2, 3, 4):
        for n in sizes:
            f = RAW / "pisinger" / f"knapPI_{t}_{n}_1000.csv"
            if not f.exists():
                continue
            fam = {1: "uncorrelated", 2: "weakly corr.", 3: "strongly corr.", 4: "inv. strongly corr."}[t]
            for inst in read_pisinger(f, limit=per_file):
                r = solve_knapsack(inst, "dp")
                ok = r.x @ inst.weights <= inst.capacity and r.x @ inst.profits == r.value
                rows.append(_row("L0 knapsack", f"Pisinger {fam}", inst.name, "DP (numba)", r.value,
                                 inst.optimum, "published optimum", "max", r.runtime,
                                 proven=True, feasible=ok, n=inst.n))
                t0 = time.perf_counter()
                v, x, nodes, proven = kp_bb(inst.profits, inst.weights, inst.capacity,
                                            node_limit=5_000_000)
                rows.append(_row("L0 knapsack", f"Pisinger {fam}", inst.name, "branch & bound",
                                 v, inst.optimum, "published optimum", "max",
                                 time.perf_counter() - t0, proven=proven, n=inst.n, nodes=nodes))
    return rows


# ------------------------------------------------------------------------------ L1
def bench_l1(quick: bool, backend: str) -> list[dict]:
    rows = []
    files = [1] if quick else list(range(1, 10))
    picks = [0, 10, 20] if quick else [0, 1, 2, 10, 11, 12, 20, 21, 22]
    limit = 5.0 if quick else 10.0
    for k in files:
        insts = read_mknap(RAW / f"mknapcb{k}.txt", RAW / "mkcbres.txt")
        for idx in picks:
            inst = insts[idx]
            fam = f"Chu-Beasley m={inst.m} n={inst.n}"
            ref = inst.best_known
            lp = mkp.solve_lp(inst, backend)
            g = mkp.greedy_dual(inst, lp.duals)
            rows.append(_row("L1 allocation (MKP)", fam, inst.name, "dual-ratio greedy + 1-swap",
                             g.value, ref, "best known", "max", g.runtime + 0.0,
                             feasible=mkp.is_feasible(inst, g.x)))
            L = mkp_lagrangian(inst, iters=60 if quick else 150, lp_duals=lp.duals)
            rows.append(_row("L1 allocation (MKP)", fam, inst.name, "Lagrangian (KP subproblem)",
                             L.value, ref, "best known", "max", L.runtime, bound=L.bound,
                             feasible=mkp.is_feasible(inst, L.x), lp_bound=lp.value,
                             bound_improvement_vs_lp=lp.value - L.bound))
            r = mkp.solve_mip(inst, backend, time_limit=limit)
            rows.append(_row("L1 allocation (MKP)", fam, inst.name, f"MIP ({backend}, {limit:g}s)",
                             r.value, ref, "best known", "max", r.runtime, bound=r.bound,
                             proven=r.status == "optimal", feasible=mkp.is_feasible(inst, r.x)))
    return rows


# ------------------------------------------------------------------------------ L2
def bench_l2(quick: bool, backend: str) -> list[dict]:
    rows = []
    files = ["gap1", "gap6", "gap12", "gapa"] if quick else \
        [f"gap{i}" for i in range(1, 13)] + ["gapa", "gapb", "gapc", "gapd"]
    limit = 5.0 if quick else 20.0
    for f in files:
        insts = read_gap(RAW / f"{f}.txt")
        if quick:
            insts = insts[:2]
        for inst in insts:
            fam = "OR-Library gap1-12" if f[3:].isdigit() else f"OR-Library {f} (type {f[-1].upper()})"
            r = gap.solve_mip(inst, backend, time_limit=limit)
            ref, kind = inst.optimum, "published optimum"
            if ref is None and r.status == "optimal":
                ref, kind = r.value, "own proven optimum"
            rows.append(_row("L2 matching (GAP)", fam, inst.name, f"MIP ({backend}, {limit:g}s)",
                             r.value, ref, kind, inst.sense, r.runtime, bound=r.bound,
                             proven=r.status == "optimal", feasible=gap.is_feasible(inst, r.assign)))
            L = gap_lagrangian(inst, iters=150 if quick else 300, time_limit=limit)
            ok = L.x.min() >= 0 and gap.is_feasible(inst, L.x)
            rows.append(_row("L2 matching (GAP)", fam, inst.name, "Lagrangian (KP per agent)",
                             L.value if ok else np.nan, ref, kind, inst.sense, L.runtime,
                             bound=L.bound, feasible=ok,
                             proven=ok and abs(L.bound - L.value) < 1 - 1e-9))
            g = gap.greedy_regret(inst)
            rows.append(_row("L2 matching (GAP)", fam, inst.name, "regret greedy + local search",
                             g.value, ref, kind, inst.sense, g.runtime,
                             feasible=np.isfinite(g.value)))
    return rows


# ------------------------------------------------------------------------------ L3
def bench_l3(quick: bool, backend: str) -> list[dict]:
    rows = []
    files = [1, 5] if quick else list(range(1, 9))
    for k in files:
        insts = read_binpack(RAW / f"binpack{k}.txt")
        if quick:
            insts = insts[:5]
        for inst in insts:
            fam = f"Falkenauer {'uniform' if inst.name.startswith('u') else 'triplet'} n={inst.n}"
            ref = inst.best_known
            kind = "proven optimum" if inst.proven_optimal else "best known"
            for name, fn in (("FFD", bpp.ffd), ("BFD", bpp.bfd)):
                r = fn(inst)
                rows.append(_row("L3 loading (BPP)", fam, inst.name, name, r.n_bins, ref, kind,
                                 "min", r.runtime, bound=r.lower_bound, proven=r.proven_optimal,
                                 feasible=bpp.is_feasible(inst, r.bins)))
            r = column_generation(inst, backend)
            rows.append(_row("L3 loading (BPP)", fam, inst.name,
                             "column generation (+dive, +arc-flow)", r.n_bins, ref, kind, "min",
                             r.runtime, bound=r.lower_bound, proven=r.proven_optimal,
                             feasible=bpp.is_feasible(inst, r.bins),
                             lp_bound=r.extra["lp_bound"], columns=r.extra["columns"],
                             used_dive=r.extra["dive"], used_arcflow=r.extra["arcflow"]))
            if inst.n <= 120 and (quick or inst.name.endswith(("_00", "_01", "_02"))):
                r = bpp.compact_mip(inst, backend, time_limit=10)
                rows.append(_row("L3 loading (BPP)", fam, inst.name, "compact assignment MIP (10s)",
                                 r.n_bins, ref, kind, "min", r.runtime, bound=r.lower_bound,
                                 proven=r.proven_optimal, feasible=bpp.is_feasible(inst, r.bins)))
    return rows


# ------------------------------------------------------------------------------ main
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["layer", "family", "method"], sort=False)
            .agg(instances=("instance", "count"),
                 matches_reference=("matches_reference", "mean"),
                 proven_optimal=("proven_optimal", "mean"),
                 mean_gap=("gap_to_reference", "mean"),
                 worst_gap=("gap_to_reference", "max"),
                 all_feasible=("feasible", "all"),
                 mean_seconds=("seconds", "mean"))
            .reset_index())


def _pct(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.3%}"


def to_markdown(s: pd.DataFrame) -> str:
    lines = ["| Layer | Instances | Method | n | = reference | proven opt. | mean gap | worst gap | always feasible | mean s |",
             "|---|---|---|---:|---:|---:|---:|---:|:---:|---:|"]
    for r in s.itertuples():
        lines.append(
            f"| {r.layer} | {r.family} | {r.method} | {r.instances} | {r.matches_reference:.0%} | "
            f"{r.proven_optimal:.0%} | {_pct(r.mean_gap)} | {_pct(r.worst_gap)} | "
            f"{'yes' if r.all_feasible else 'no'} | {r.mean_seconds:.2f} |"
        )
    return "\n".join(lines) + "\n"


def run(quick: bool = False, backend: str = "highs", out: Path = ROOT / "results") -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, fn in (("L0", lambda: bench_l0(quick)), ("L1", lambda: bench_l1(quick, backend)),
                     ("L2", lambda: bench_l2(quick, backend)), ("L3", lambda: bench_l3(quick, backend))):
        t0 = time.perf_counter()
        part = fn()
        rows += part
        print(f"{name}: {len(part)} rows in {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    suffix = "_quick" if quick else ""
    df.to_csv(out / f"benchmarks{suffix}.csv", index=False)
    s = summarise(df)
    (out / f"benchmarks{suffix}.md").write_text(to_markdown(s))
    print(to_markdown(s))
    return df


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
