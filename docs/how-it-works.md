# How it works

Each day, vinpack runs a three-layer pipeline and every layer is a knapsack-family problem:

| Layer | Planning stage | Problem | Exact method | Fast method |
|---|---|---|---|---|
| **L1 Allocation** | Build plan / firming: *which orders get built this cycle?* | Multidimensional knapsack: orders compete for 10 scarce supply pools | MIP (HiGHS or Gurobi) | Dual-ratio greedy; Lagrangian relaxation with a knapsack subproblem |
| **L2 Matching** | Vehicle plan: *which delivery center fulfils each order?* | Generalized assignment: every filled order goes to one DC, and DCs have handling capacity | MIP | Lagrangian relaxation (one knapsack per DC); regret greedy |
| **L3 Loading** | Routing: *how many carriers leave each DC?* | Bin packing: order load units onto carriers of capacity 150 | Column generation, then an arc-flow MIP if needed | FFD / BFD |
| **L0 Kernel** | — | 0-1 knapsack | Numba DP (float profits) | Branch and bound |

The **L0 kernel is reused three times**: as the column-generation pricing problem in L3, as the
Lagrangian subproblem in L1, and as the per-DC subproblem in the L2 Lagrangian.

## Re-solving every day without thrashing the plan

The plan is re-solved every day as orders arrive, customers cancel, supply drifts and a DC loses capacity.
A plan that is re-optimised from scratch each day moves customers around constantly. So each daily
solve penalises changes to yesterday's decisions:

$$\max \sum_j p_j x_j \;-\; \lambda \sum_{j \in \text{yesterday}} |x_j - \bar x_j|$$

Because $x$ is binary, $|x_j - \bar x_j|$ is linear: it equals $x_j$ where $\bar x_j = 0$ and $1 - x_j$
where $\bar x_j = 1$. **No extra variables or constraints are needed.** Orders filled for three consecutive
days count as *promised* and carry a large soft lock. It is soft so that a supply shock can still
break a promise instead of making the model infeasible, and every broken promise is counted. L2
uses the same idea for DC reassignments. Sweeping λ traces the **value-vs-stability frontier**.

## Explaining the plan

The explainer can only cite facts computed by the solver:

* **LP shadow prices** of each supply pool: what one more unit is worth.
* **Reduced cost** of each order: its value minus the shadow cost of the supply it would use. A
  negative reduced cost means the order is *priced out*.
* **What-if re-solves** of the exact same day: force an order in, or add supply. The day is
  rebuilt deterministically from the seed and yesterday's stored plan. Each what-if reports the
  value change, the churn change and the churn-adjusted objective change, so a what-if that buys
  value by reshuffling yesterday's plan is not mistaken for a free win.

The Claude agent (`claude-opus-5`, set by `VINPACK_MODEL`) calls these as tools. After it
answers, every number in the answer is checked against the tool outputs, and numbers without a
source are flagged. What-ifs are **proposals**: nothing changes until a planner clicks *Approve*.
Approving replays the run from that day into a new run id, so the original run is never
overwritten.

## Where the data comes from

Every number comes from published academic benchmark files, fetched and checked against a SHA-256
manifest:

* **L1**: OR-Library `mknapcb5` problem 10.250-20 (Chu & Beasley 1998): 250 orders, 10 pools, tightness 0.75.
* **L2**: OR-Library `gapc` 10x200-4 plus the first 50 jobs of `gapc` 10x100-3. Type-C capacity is
  defined as $b_i = 0.8 \sum_j r_{ij} / m$. That formula is additive over jobs, so it is recomputed
  exactly on the concatenated columns and then scaled by the expected fill rate.
* **L3**: Falkenauer `u250_00` item sizes (OR-Library `binpack2`), carrier capacity 150.

The composed scenario is **not** a published benchmark. Each layer is also benchmarked on its own
against published optima on the **unmodified** instances. See [Results](results.md).
