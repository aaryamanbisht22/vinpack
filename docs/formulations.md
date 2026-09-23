# Formulations

Notation: orders $j$, supply pools $i$, delivery centers (agents) $a$, carriers $k$.

## L1: Allocation (multidimensional knapsack)

$$
\max \sum_j p_j x_j \quad \text{s.t.} \quad \sum_j r_{ij} x_j \le b_i \;\; \forall i, \qquad x_j \in \{0,1\}
$$

**Lagrangian with a knapsack subproblem.** Keep the pool $k$ that carries the most LP value, and relax
the others with multipliers $\lambda \ge 0$:

$$
L(\lambda) = \sum_{i \ne k} \lambda_i b_i + \max_{x \in \{0,1\}^n} \Big\{ \sum_j \big(p_j - \textstyle\sum_{i\ne k} \lambda_i r_{ij}\big) x_j : \sum_j r_{kj} x_j \le b_k \Big\}
$$

The inner problem is a 0-1 knapsack, solved exactly by the L0 DP. It lacks the integrality property,
so $\min_\lambda L(\lambda)$ can be strictly tighter than the LP bound. The benchmark reports how much
tighter. Multipliers start at the LP duals and are updated by subgradient steps with a Polyak step size.

## L2: Matching (generalized assignment)

$$
\min \sum_{a,j} c_{aj} y_{aj} \quad \text{s.t.} \quad \sum_a y_{aj} = 1 \;\; \forall j, \qquad \sum_j r_{aj} y_{aj} \le b_a \;\; \forall a, \qquad y \in \{0,1\}
$$

**Lagrangian (Ross–Soland / Fisher–Jaikumar–Van Wassenhove).** Relax the assignment rows with
free multipliers $u_j$. The problem decomposes into one 0-1 knapsack per DC, each solved by L0. A
repair heuristic (keep the best agent, place the rest by regret, then shift/swap local search)
turns each relaxed solution into a feasible assignment.

## L3: Loading (bin packing)

**Gilmore–Gomory column generation.** Items of equal size are grouped into types $t$ with demand
$d_t$. A column is a feasible carrier load $a_t$.

$$
\text{Master LP: } \min \sum_p \lambda_p \;\text{ s.t. }\; \sum_p a_{tp} \lambda_p \ge d_t, \quad \lambda \ge 0
\qquad
\text{Pricing: } \max \sum_t \pi_t a_t \;\text{ s.t. }\; \sum_t s_t a_t \le C
$$

The pricing problem is a bounded knapsack. It is binary-split into 0-1 copies and solved by the L0 DP
with the **float dual prices as profits**. $\lceil z^*_{LP} \rceil$ is a valid lower bound, and it is
tight on almost every practical instance (the IRUP property). Once an integer plan reaches that bound,
it is **proven optimal**.

Integer phase, stopping as soon as the bound is reached:

1. round the LP down, then FFD the residual items;
2. solve the restricted master as an IP over the generated columns;
3. **residual diving**: fix the columns with $\lambda \ge 1$ (or the largest one), keep only patterns that
   fit the remaining demand, re-price, and repeat;
4. **arc-flow MIP** (Valério de Carvalho 1999), capped at the incumbent: nodes are partial loads
   $0..C$, and item arcs of type $t$ leave only nodes reachable by larger types.

The Falkenauer *triplet* instances are the hard case: every optimal carrier is exactly full, so
one rounding mistake costs a whole carrier. Steps 3–4 are what close them.

## L0: The knapsack kernel

* **DP** over capacity with a bit-table for reconstruction, $O(nC)$ time, float profits.
* **Branch and bound** (Horowitz–Sahni) with the Dantzig bound. It does not depend on $C$, but it
  blows up on Pisinger's strongly-correlated instances, which is exactly what Pisinger's
  *Where are the hard knapsack problems?* predicts. The benchmark shows this directly.

## Stability term

For binary $x$ and yesterday's decision $\bar x$, $|x_j - \bar x_j| = x_j$ if $\bar x_j = 0$ and $1 - x_j$ if
$\bar x_j = 1$. The churn penalty is therefore a change to the objective coefficients plus a constant
offset. It adds no variables and no constraints, so the daily model stays as easy to solve as the
static one.
