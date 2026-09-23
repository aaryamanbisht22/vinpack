"""vinpack dashboard: daily plan, plan diffs, ask-why agent with approval, stability frontier, benchmarks.

    uv run vinpack app          (or: uv run streamlit run app/streamlit_app.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vinpack import store  # noqa: E402
from vinpack.explain import agent, templates  # noqa: E402
from vinpack.explain.evidence import Evidence  # noqa: E402
from vinpack.pipeline.adapter import load_default  # noqa: E402
from vinpack.pipeline.run import DayParams  # noqa: E402
from vinpack.simulate.engine import replay_from, run_and_store  # noqa: E402
from vinpack.simulate.scenarios import WorldConfig  # noqa: E402

# Categorical slots 1-3 of the validated reference palette (all-pairs safe), plus ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#8a8985"

st.set_page_config(page_title="vinpack - allocation, matching, loading", layout="wide")


# Files the demo scenario is built from (see pipeline/adapter.py).
SCENARIO_FILES = ["mknapcb5.txt", "mkcbres.txt", "gapc.txt", "binpack2.txt"]
PARETO_WEIGHTS = [0, 0.1, 0.25, 0.5, 1, 2]


@st.cache_resource
def ensure_data() -> None:
    """On a fresh host (e.g. Streamlit Community Cloud), download the OR-Library files once."""
    from vinpack.io import fetch

    if all((fetch.RAW / f).exists() for f in SCENARIO_FILES):
        return
    fetch.RAW.mkdir(parents=True, exist_ok=True)
    with st.spinner("First launch: downloading benchmark data from OR-Library ..."):
        fetch.fetch_orlib()
    bad = fetch.verify_manifest()
    if bad:
        st.error(f"Downloaded files failed their SHA-256 check: {bad}")
        st.stop()


@st.cache_resource
def scenario():
    ensure_data()
    return load_default()


def seed_demo_runs(con, scn) -> None:
    """First launch with an empty store: solve the default run and the stability sweep."""
    with st.status("First launch: solving the demo runs (about a minute) ...", expanded=True) as s:
        st.write("14-day rolling re-solve, churn weight 0.5")
        run_and_store(con, "default", scn, WorldConfig(days=14), DayParams(churn_weight=0.5))
        for w in PARETO_WEIGHTS:
            st.write(f"stability sweep: churn weight {w:g}")
            run_and_store(con, f"pareto-{w:g}", scn, WorldConfig(days=14),
                          DayParams(churn_weight=w, l1_time=3, l2_time=3))
        s.update(label="Demo runs ready", state="complete")


# One connection per script run, closed at the end of the run, so the CLI (simulate,
# pareto, publish) can write to the same DuckDB file while the dashboard is open.
_CON = store.connect()


def cur():
    return _CON


def fig_layout(fig: go.Figure, title: str, y_title: str, height: int = 300) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height,
        margin=dict(l=10, r=10, t=45, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, title="day", dtick=1)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, title=y_title)
    return fig


def line(fig, x, y, name, color, dash=None):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers",
                             line=dict(color=color, width=2, dash=dash), marker=dict(size=8)))


scn = scenario()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("vinpack")
st.sidebar.caption("Allocation -> matching -> loading, re-solved daily. "
                   "Every number traces to an academic benchmark instance.")
runs = store.runs(cur())
if runs.empty:
    seed_demo_runs(cur(), scn)
    runs = store.runs(cur())
with st.sidebar.expander("New simulation run", expanded=runs.empty):
    new_id = st.text_input("run id", "default")
    churn = st.slider("churn weight (x mean order value)", 0.0, 2.0, 0.5, 0.05)
    days = st.slider("days", 3, 21, 14)
    seed = st.number_input("seed", value=7, step=1)
    if st.button("Run simulation", type="primary"):
        with st.spinner("Solving each day: MKP -> GAP -> column generation ..."):
            run_and_store(cur(), new_id, scn, WorldConfig(days=days, seed=int(seed)),
                          DayParams(churn_weight=churn))
        st.rerun()
if runs.empty:
    _CON.close()
    st.stop()
run_ids = runs["run_id"].tolist()
run_id = st.sidebar.selectbox("run", run_ids,
                              index=run_ids.index("default") if "default" in run_ids else 0)
row = runs.set_index("run_id").loc[run_id]
st.sidebar.write(f"churn weight **{row.churn_weight:g}**, seed {row.seed}, {row.days} days, "
                 f"backend `{row.backend}`")
with st.sidebar.expander("Data sources"):
    for k, v in scn.sources.items():
        st.write(f"**{k}**: {v}")

kpis = store.table(cur(), "kpis", run_id)
tab_plan, tab_diff, tab_ask, tab_pareto, tab_bench, tab_about = st.tabs(
    ["Daily plan", "Plan diff", "Ask why", "Stability frontier", "Benchmarks", "How it works"])

# ---------------------------------------------------------------- daily plan
with tab_plan:
    c = st.columns(5)
    c[0].metric("Plan value (horizon)", f"{kpis.value.sum():,.0f}")
    c[1].metric("Fill rate (last day)", f"{kpis.filled.iloc[-1] / kpis.open.iloc[-1]:.0%}")
    c[2].metric("Orders re-planned", int(kpis.churn_l1.sum()),
                help="Orders whose filled/unfilled status flipped vs the previous day")
    c[3].metric("Days with optimal loading",
                f"{int((kpis.trucks == kpis.trucks_lb).sum())}/{len(kpis)}",
                help="Trucks used equals the column-generation / arc-flow lower bound")
    c[4].metric("Worst L1 MIP gap", f"{kpis.l1_gap.max():.2%}")

    left, right = st.columns(2)
    f = go.Figure()
    line(f, kpis.day, kpis.open, "open orders", MUTED, "dot")
    line(f, kpis.day, kpis.filled, "filled", BLUE)
    left.plotly_chart(fig_layout(f, "Orders open vs filled", "orders"), use_container_width=True)
    f = go.Figure()
    line(f, kpis.day, kpis.churn_l1, "fill decisions changed", BLUE)
    line(f, kpis.day, kpis.churn_l2, "DC reassignments", ORANGE)
    line(f, kpis.day, kpis.broken_promises, "broken promises", AQUA)
    right.plotly_chart(fig_layout(f, "Plan churn vs yesterday", "orders"), use_container_width=True)

    day = st.slider("Inspect day", int(kpis.day.min()), int(kpis.day.max()), int(kpis.day.max()))
    ev_rows = store.table(cur(), "events", run_id, day)
    for e in ev_rows.event:
        st.caption(f"Event: {e}")
    pools = store.table(cur(), "pools", run_id, day)
    dec = store.table(cur(), "decisions", run_id, day)
    left, right = st.columns(2)
    util = (pools.used / pools.capacity).to_numpy()
    f = go.Figure(go.Bar(
        y=pools.name, x=util, orientation="h", marker=dict(color=BLUE, cornerradius=4),
        customdata=np.c_[pools.used, pools.capacity, pools.shadow_price],
        hovertemplate="%{y}<br>used %{customdata[0]:,.0f} / %{customdata[1]:,.0f}"
                      "<br>shadow price %{customdata[2]:.4f}<extra></extra>"))
    fig_layout(f, f"Supply pool utilisation, day {day}", "", 360)
    f.update_xaxes(title="share of supply used", tickformat=".0%", dtick=None, range=[0, 1.05])
    f.update_layout(hovermode="closest")
    left.plotly_chart(f, use_container_width=True)
    left.caption("Pools with a positive shadow price are binding: one more unit is worth that "
                 "much plan value. Hover a bar for the numbers.")

    ev = Evidence(cur(), run_id, scn)
    bc = ev.binding_constraints(day)
    dcs = pd.DataFrame(bc["delivery_centers"])
    f = go.Figure(go.Bar(
        y=dcs.dc, x=dcs.utilisation, orientation="h", marker=dict(color=ORANGE, cornerradius=4),
        customdata=np.c_[dcs.used, dcs.capacity],
        hovertemplate="%{y}<br>load %{customdata[0]} / capacity %{customdata[1]}<extra></extra>"))
    fig_layout(f, f"Delivery-center utilisation, day {day}", "", 360)
    f.update_xaxes(title="share of handling capacity used", tickformat=".0%", dtick=None)
    f.update_layout(hovermode="closest")
    right.plotly_chart(f, use_container_width=True)

    with st.expander(f"Orders on day {day}"):
        view = dec[dec.is_open].copy()
        view["dc"] = [scn.agent_names[a] if a >= 0 else "" for a in view.agent]
        st.dataframe(view[["order_id", "value", "filled", "dc", "truck", "streak", "reduced_cost"]],
                     hide_index=True, use_container_width=True)
    with st.expander("Day-level KPI table"):
        st.dataframe(kpis.drop(columns=["run_id"]), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- plan diff
with tab_diff:
    d = st.slider("Day", 1, int(kpis.day.max()), min(9, int(kpis.day.max())), key="diffday")
    diffs = store.table(cur(), "diffs", run_id, d)
    if diffs.empty:
        st.info("No changes on this day.")
    else:
        counts = diffs.change.value_counts()
        cols = st.columns(len(counts))
        for col, (k, v) in zip(cols, counts.items(), strict=True):
            col.metric(k.replace("_", " "), int(v))
        kinds = st.multiselect("change types", counts.index.tolist(), counts.index.tolist())
        show = diffs[diffs.change.isin(kinds)].copy()
        show["from"] = [scn.agent_names[a] if a >= 0 else "" for a in show.from_agent]
        show["to"] = [scn.agent_names[a] if a >= 0 else "" for a in show.to_agent]
        st.dataframe(show[["order_id", "change", "from", "to", "reason"]], hide_index=True,
                     use_container_width=True)

# ---------------------------------------------------------------- ask why
with tab_ask:
    st.write("Ask about any order or day. The agent can only cite solver facts (shadow prices, "
             "reduced costs, capacities, what-if re-solves). Proposals change nothing until you "
             "approve them.")
    c1, c2 = st.columns([1, 1])
    a_day = c1.number_input("day", 0, int(kpis.day.max()), min(9, int(kpis.day.max())))
    a_order = c2.number_input("order id", 0, scn.n_orders - 1, 114)
    q = st.text_input("question", f"Why isn't order {a_order} filled on day {a_day}, "
                                  "and what would it take?")
    use_claude = agent.has_credentials()
    st.caption(f"Explainer: {'Claude (' + agent.MODEL + ')' if use_claude else 'deterministic template - set ANTHROPIC_API_KEY to enable the Claude agent'}")
    if st.button("Explain", type="primary"):
        ev = Evidence(cur(), run_id, scn)
        with st.spinner("Reading the plan and running what-ifs ..."):
            if use_claude:
                try:
                    ans = agent.ask(ev, q)
                    st.session_state["answer"] = ans
                except Exception as e:  # network/auth: fall back, but say so
                    st.warning(f"Claude call failed ({type(e).__name__}); using the template.")
                    st.session_state["answer"] = agent.AgentAnswer(
                        templates.explain_order(ev, int(a_day), int(a_order)))
            else:
                st.session_state["answer"] = agent.AgentAnswer(
                    templates.explain_order(ev, int(a_day), int(a_order)))
        # keep only plain data across reruns (the connection closes after each run)
        st.session_state["proposals"] = {"run_id": run_id, **ev.proposals}
    ans = st.session_state.get("answer")
    if ans:
        st.markdown(ans.text)
        if ans.unverified:
            st.warning(f"Numbers not found in any tool output: {', '.join(ans.unverified)}")
        if ans.tool_calls:
            with st.expander(f"{len(ans.tool_calls)} tool calls"):
                for call in ans.tool_calls:
                    st.write(f"`{call['tool']}` {call['input']}")
                    st.json(call["output"], expanded=False)
        props = dict(st.session_state.get("proposals", {}))
        if props.pop("run_id", None) == run_id and props:
            st.subheader("Proposals awaiting approval")
            for pid, prop in props.items():
                st.write(f"**{pid}** - day {prop['day']}: `{prop['action']}`")
                if st.button(f"Approve {pid}: replay run from day {prop['day']}", key=pid):
                    new_run = f"{run_id}+{pid}"
                    with st.spinner("Re-solving the remaining days with the override ..."):
                        replay_from(cur(), run_id, new_run, prop["day"], prop["params"], scn)
                    st.success(f"Applied as new run '{new_run}' (select it in the sidebar). "
                               "The original run is unchanged.")

# ---------------------------------------------------------------- pareto
with tab_pareto:
    st.write("Each point is a full 14-day simulation at one churn weight λ. Raising λ moves "
             "left (fewer orders re-planned day to day) and down (less plan value): the "
             "curve is the price of a stable plan.")
    pr = runs[runs.run_id.str.startswith("pareto-")]
    if st.button("Run sweep (6 weights)"):
        with st.spinner("Running 6 simulations ..."):
            for w in [0, 0.1, 0.25, 0.5, 1, 2]:
                run_and_store(cur(), f"pareto-{w:g}", scn, WorldConfig(days=14),
                              DayParams(churn_weight=w, l1_time=3, l2_time=3))
        st.rerun()
    if pr.empty:
        st.info("No sweep stored yet. Run it here or with `vinpack pareto`.")
    else:
        rows = []
        for r in pr.itertuples():
            k = store.table(cur(), "kpis", r.run_id)
            rows.append({"churn weight": r.churn_weight, "total value": k.value.sum(),
                         "orders re-planned": k.churn_l1.sum(),
                         "DC reassignments": k.churn_l2.sum(),
                         "broken promises": k.broken_promises.sum(), "trucks": k.trucks.sum()})
        pf = pd.DataFrame(rows).sort_values("churn weight")
        f = go.Figure(go.Scatter(
            x=pf["orders re-planned"], y=pf["total value"], mode="lines+markers+text",
            text=[f"λ={w:g}" for w in pf["churn weight"]], textposition="top right",
            line=dict(color=BLUE, width=2), marker=dict(size=10),
            hovertemplate="re-planned %{x}<br>value %{y:,.0f}<extra>%{text}</extra>"))
        fig_layout(f, "Value vs plan stability", "total plan value (14 days)", 380)
        f.update_xaxes(title="orders re-planned (sum over days)", dtick=None)
        f.update_layout(hovermode="closest")
        st.plotly_chart(f, use_container_width=True)
        st.dataframe(pf, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- benchmarks
with tab_bench:
    path = ROOT / "results" / "benchmarks.csv"
    if not path.exists():
        path = ROOT / "results" / "benchmarks_quick.csv"
    if not path.exists():
        st.info("Run `vinpack bench` (or `vinpack bench --quick`) to populate this tab.")
    else:
        b = pd.read_csv(path)
        st.write("Every algorithm on the **unmodified** academic instances, against published "
                 "optima / best-known values.")
        summary = (b.groupby(["layer", "family", "method"])
                   .agg(instances=("instance", "count"),
                        matches_reference=("matches_reference", "mean"),
                        proven_optimal=("proven_optimal", "mean"),
                        mean_gap_to_reference=("gap_to_reference", "mean"),
                        mean_seconds=("seconds", "mean"))
                   .reset_index())
        st.dataframe(summary.style.format({"matches_reference": "{:.0%}", "proven_optimal": "{:.0%}",
                                           "mean_gap_to_reference": "{:.3%}",
                                           "mean_seconds": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        with st.expander("All rows"):
            st.dataframe(b, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown((ROOT / "docs" / "how-it-works.md").read_text()
                if (ROOT / "docs" / "how-it-works.md").exists() else "See README.md.")

_CON.close()
