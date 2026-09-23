# Study guide: how to explain every part of vinpack

This guide is for talking about the project in an interview. Read it top to bottom once, run the
commands next to each section, and then practise the questions at the end out loud.

---

## 1. The 30-second story

> "Car makers have limited parts, so each week they must choose which customer orders to build.
> Each car then goes to a delivery center, and the cars leave on carriers. I built a daily planner
> that makes all three decisions with optimisation. Each step is a variant of the knapsack
> problem. Every morning it re-plans as orders arrive, customers cancel and a delivery center
> loses capacity, but it penalises changes to yesterday's plan so customers aren't shuffled
> around. I tested each step on published academic benchmark sets where the best answers are
> known. It matches all 60 published optima on the assignment set, and it proves the optimal
> number of carriers on all 160 bin-packing instances."

---

## 2. The one idea everything is built on: the knapsack

You have a bag that holds 10 kg, and items with a weight and a value. Pick the items that give the
most value without going over 10 kg. Each item is either in (1) or out (0).

- **L1 (allocation)** is the same problem with **10 bags at once**. Each order uses some battery
  packs, some drive units, some paint hours and so on, and every supply pool has a limit. This is
  the *multidimensional knapsack*.
- **L2 (matching)** gives each delivery center its own bag (its handling capacity). Every order must
  go into exactly one bag, and each order costs a different amount depending on which bag it goes
  in. This is the *generalized assignment problem*.
- **L3 (loading)** makes every carrier a bag of size 150, and asks for the fewest bags that hold
  everything. This is *bin packing*.
- **L0** is the plain one-bag knapsack. L3 calls it repeatedly (see column generation below).

---

## 3. File by file

### `solvers/kp_dp.py`: the one-bag knapsack (L0)

**Idea: dynamic programming.** `best[c]` means "the most value I can get with capacity c using
the items seen so far". For each item, for each capacity c from high to low:
`best[c] = max(best[c], best[c - weight] + value)`. Going from high to low stops you using the same
item twice. A second table records which items were taken, so the chosen set can be rebuilt at the end.

**Tiny example:** capacity 5. Item A (weight 2, value 3), item B (weight 3, value 4). After A,
best[2..5] = 3. After B, best[5] = max(3, best[2] + 4) = 7, so take both.

**Speed:** O(n × C). It's compiled with **Numba**, which turns the Python loop into machine code.

**Branch and bound** is the second method in the file. It explores "take item / skip item" as a
tree and prunes any branch whose optimistic estimate can't beat the best answer found so far. On
Pisinger's *strongly correlated* instances (value ≈ weight + constant) the estimates barely prune
anything, so it runs out of time. That is the result Pisinger's paper is known for, and it's why
the DP is used.

Try it: `make test`, then look at `tests/test_knapsack.py`.

### `solvers/backend.py`: talking to the solver

Every model is written as matrices: an objective vector `c`, a constraint matrix `A` with row
lower and upper bounds, variable bounds, and which variables must be integers. The file passes
those arrays to **HiGHS**, a free solver, and gets back the solution, a bound and (for LPs) the
**dual values**.

**Dual value / shadow price:** how much the objective would go up if you had one more unit of a
resource. `tests/test_mkp.py::test_duals_are_marginal_values` checks this literally: it adds one
unit of a pool, re-solves, and confirms the objective went up by exactly the dual.

### `solvers/mkp.py`: which orders to fill (L1)

- `build_model` writes the multidimensional knapsack as matrices.
- `solve_mip` solves it exactly with integer variables.
- `solve_lp` solves the **LP relaxation** (x may be fractional, e.g. 0.4 of an order). It's
  easier to solve and gives the shadow price of each supply pool.
- `greedy_dual` is the fast heuristic. Score each order by value ÷ (shadow-price-weighted supply it
  uses), fill in score order, then try swapping one order for another.

**Reduced cost** = an order's value minus the shadow cost of the supply it uses. If it's negative,
the order is "priced out": the supply it needs is worth more to other orders. The explainer uses
this.

### `solvers/gap.py`: which delivery center (L2)

Variables y[a, j] = 1 if order j goes to delivery center a. There are two constraint types: each
order is assigned exactly once, and each center stays within capacity. `solve_mip` is exact.
`greedy_regret` is the heuristic. It first places the order that would lose the most by not getting
its best center (its "regret"), then improves the result by moving and swapping orders.

### `solvers/bpp.py`, `colgen.py`, `arcflow.py`: loading carriers (L3)

- **FFD (first-fit decreasing):** sort items big to small and put each one in the first carrier
  it fits in. It's fast and usually good, **but it can never prove it's optimal**.
- **Lower bound:** you can't use fewer carriers than the total size divided by 150, rounded up.
  If a plan uses exactly that many, it is **proven** optimal.
- **Column generation (`colgen.py`):** instead of deciding item by item, choose from possible
  *carrier loads* ("patterns"). There are far too many patterns to list, so it works like this:
  1. Start with a few patterns (from FFD).
  2. Solve an LP that picks how many of each pattern to use. Its duals say how valuable each item
     type is right now.
  3. Ask the L0 knapsack for the most valuable single carrier load at those prices. If that load is
     worth more than 1 carrier, add it as a new pattern and repeat. If not, the LP is optimal.
  4. That LP value, rounded up, is a very strong lower bound.
- **Getting whole numbers back:** round down the LP and FFD whatever is left over. If that misses
  the bound, "dive" (fix the most-used patterns and re-solve what remains). If that still misses,
  run the **arc-flow** model: a different exact MIP where each carrier is a path from load 0 to
  load 150. Arc-flow is what closes the "triplet" instances, where every optimal carrier is exactly
  full and one bad rounding decision costs a whole carrier.

### `pipeline/adapter.py`: turning benchmark files into one scenario

It takes one real instance per layer from the benchmark files and assigns business roles: the
knapsack rows become supply pools, the GAP agents become delivery centers, and the bin-packing
sizes become carrier load units. The names ("Chicago DC", "battery packs") are labels. The numbers
are the published data.

### `pipeline/run.py`: one day of planning

`solve_day` runs L1, then L2 on the filled orders, then L3 per delivery center, and records KPIs.

**The churn penalty (the most important idea to explain well):** yesterday order j was filled
(x̄ = 1) or not (x̄ = 0). A change today costs λ. For a 0/1 variable, |x − x̄| is just x (if
x̄ = 0) or 1 − x (if x̄ = 1). That is plain linear arithmetic, so the penalty only adjusts each
order's objective coefficient. **No new variables and no new constraints**, so the daily model
is no harder than the one-off model.

**Promises:** an order filled 3 days in a row is "promised" and gets a very large penalty for being
dropped. It's a *soft* lock. A hard constraint could make the model impossible to solve after a
supply shock, whereas a large penalty always yields a plan, and any broken promise gets counted.

### `simulate/`: the fake world and the daily loop

- `scenarios.py`: a random but **reproducible** world (same seed, same days). 60% of orders exist
  on day 0 and the rest arrive later. About 1% cancel per day, supply drifts a few percent, and
  Chicago loses 35% of its capacity for 3 days mid-way.
- `engine.py`: loops over the days, feeding each day's plan into the next. `replay_from` re-runs
  from a given day with a planner override and saves it as a new run.
- `diff.py`: compares today with yesterday and writes a reason for each change, taken from
  solver facts (shadow prices, reduced costs, capacity cuts).

### `store.py`: saving results

A DuckDB database file, which is like SQLite but built for analysis. Re-running the same day
replaces its rows instead of duplicating them ("idempotent"), so a failed day can simply be re-run.

### `explain/`: answering "why?"

- `evidence.py` gathers facts about an order (value, reduced cost, which pool is most expensive)
  and runs **what-ifs**: re-solve the same day with the order forced in and report what changes.
- `templates.py` turns those facts into a paragraph. **This is the default and needs no API key.**
- `agent.py` is optional. If an Anthropic key is set, a small Claude model (Haiku) can call the
  same evidence functions as tools. Afterwards, every number in its answer is checked against
  the tool outputs, and anything without a source is flagged.

### `app/streamlit_app.py`: the dashboard

It has tabs for the daily plan, what changed, ask-why (with an Approve button for proposals), the
value-vs-stability curve, and the benchmark table.

---

## 4. Numbers to remember

| Claim | Number |
|---|---|
| Published GAP optima matched (gap1–12) | **60 / 60** |
| Bin-packing instances proven optimal | **160 / 160**, including 4 where the file's "best known" was beaten by one carrier |
| How much worse FFD is on bin packing | 1.2% (uniform) to 16% (triplets) |
| Allocation MIP vs best-known, 10 s | 0.002–0.13% mean gap |
| Knapsack DP vs Pisinger optima | 480 / 480 |
| Churn weight 0.25 vs re-planning from scratch | −68% re-planned orders for −1.4% value |

---

## 5. Practice questions (answer out loud)

1. **What is a shadow price?** The increase in the objective from one more unit of a
   resource. I checked it by adding a unit and re-solving.
2. **Why is the churn penalty linear?** For a 0/1 variable, |x − x̄| equals x or 1 − x, so it only
   changes objective coefficients. No new variables or constraints.
3. **Why a soft lock instead of a hard constraint for promises?** A hard lock can make the model
   infeasible after a supply shock. A penalty always gives a plan and lets me count broken promises.
4. **Why can't FFD prove optimality?** It has no lower bound. Column generation gives the LP bound,
   and when the integer plan matches it, the plan is proven optimal.
5. **What does the knapsack have to do with column generation?** Each iteration asks for the most
   valuable carrier load at the current dual prices, and that is a knapsack problem.
6. **Why did you use a DP instead of branch and bound?** B&B stalls on strongly correlated
   instances, which Pisinger's benchmark shows and which my benchmark reproduces. The DP is O(nC)
   and exact for these capacities.
7. **What's a reduced cost, and how do you use it?** Value minus the shadow cost of the resources
   used. A negative one means the order is priced out. The explainer reports it.
8. **How do you know your results are correct?** Tests assert the published optima. The benchmark
   compares against published values. Every solution is checked for feasibility.
9. **What was the hardest bug?** The triplet bin-packing files use decimals like 26.3, and the
   parser was truncating them to 26, which made a solver appear to beat a proven optimum. The fix
   was to scale sizes by 10 to keep them exact integers. *Lesson: a result that looks too good is a
   bug until proven otherwise.*
10. **Another bug?** In the storage code, a DataFrame named `pools` shared its name with the `pools` table, so DuckDB inserted
    the table into itself and saved no shadow prices. The explainer crashed on day 9, which is
    how it was caught.
11. **Tell me about the what-if that looked like a win.** Forcing order 114 in raised value by 871,
    but it caused 3 extra plan changes, and after the churn penalty the objective was 254 *worse*.
    The report now shows both numbers so a planner isn't misled.
12. **How would this scale to hundreds of thousands of cars?** Split allocation by region or model
    line. Use heuristics or decomposition for matching instead of the exact MIP. Column generation
    already scales well, since each pricing step is one fast knapsack.
13. **Why HiGHS and not Gurobi?** HiGHS is free, so anyone can clone and run the project. The models
    are plain matrices, so another solver is one extra function in `backend.py`.
14. **Why DuckDB?** It's a file-based analytical database with no server, and it's fast for the
    per-day tables. Writes are idempotent per (run, day), so a failed day can be re-run.
15. **Where is the benchmark data from?** OR-Library (Beasley) and Pisinger's knapsack
    instances, all public and published. It's downloaded and checked with SHA-256 checksums.

---

## 6. How to talk about using AI

Be direct about it. The job flyer asks for *"fluent, skeptical use of AI tools"*. A good way to say it:

> "I built this with Claude Code as a pair programmer. I chose the problem, the layers and the
> benchmark sets. I directed the build, reviewed the output and verified it against published
> optima. Some of what I learned came from catching its mistakes, like the parser truncating
> decimals and a what-if that looked like a win until the churn penalty was counted. I also
> simplified it to the parts I can explain and defend."

The git history credits Claude as a co-author. Leave it that way; it matches this story.
