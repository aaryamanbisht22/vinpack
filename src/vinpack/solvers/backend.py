"""MILP layer: every exact model is one matrix-form ``LinearModel`` solved by HiGHS.

Each model in vinpack is built as a ``LinearModel`` (objective vector, sparse
constraint matrix with row ranges, column bounds, integrality) and handed to
``solve``. Keeping models in plain matrix form keeps the solver swappable: another
MILP solver (e.g. Gurobi) would be one more branch in ``solve`` that reads the
same arrays.

Dual convention: ``Solution.row_duals[i]`` is d(objective)/d(rhs_i) for the
LP relaxation, i.e. the marginal value of one more unit of row i's binding side,
in the model's own sense. Verified by a finite-difference test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

Backend = Literal["highs"]
INF = float("inf")


@dataclass
class LinearModel:
    c: np.ndarray
    A: sp.csr_matrix
    row_lb: np.ndarray
    row_ub: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray
    integer: np.ndarray  # bool per column
    sense: Literal["max", "min"] = "max"
    obj_offset: float = 0.0
    row_names: list[str] | None = None

    @property
    def n_cols(self) -> int:
        return len(self.c)

    @property
    def n_rows(self) -> int:
        return self.A.shape[0]

    def relaxed(self) -> LinearModel:
        return LinearModel(
            self.c, self.A, self.row_lb, self.row_ub, self.col_lb, self.col_ub,
            np.zeros_like(self.integer), self.sense, self.obj_offset, self.row_names,
        )


@dataclass
class Solution:
    status: str  # "optimal" | "time_limit" | "infeasible" | "other"
    objective: float
    x: np.ndarray
    bound: float  # best proven bound on the optimum (== objective for LPs)
    runtime: float
    backend: str
    row_duals: np.ndarray | None = None  # LP only
    reduced_costs: np.ndarray | None = None  # LP only
    extra: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        if not np.isfinite(self.objective) or not np.isfinite(self.bound):
            return INF
        return abs(self.bound - self.objective) / max(1.0, abs(self.objective))

    @property
    def feasible(self) -> bool:
        return self.status in ("optimal", "time_limit") and self.x is not None and len(self.x) > 0


def solve(
    model: LinearModel,
    backend: Backend = "highs",
    time_limit: float = 60.0,
    mip_gap: float = 1e-6,
    relax: bool = False,
    warm_start: np.ndarray | None = None,
    threads: int | None = None,
    verbose: bool = False,
) -> Solution:
    if relax:
        model = model.relaxed()
    if backend == "highs":
        return _solve_highs(model, time_limit, mip_gap, warm_start, threads, verbose)
    raise ValueError(f"unknown backend {backend!r} (only 'highs' is built in)")


# --------------------------------------------------------------------------- HiGHS
def _solve_highs(model, time_limit, mip_gap, warm_start, threads, verbose) -> Solution:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(verbose))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    if threads:
        h.setOptionValue("threads", int(threads))

    A = model.A.tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = model.n_cols
    lp.num_row_ = model.n_rows
    lp.col_cost_ = np.asarray(model.c, dtype=float)
    lp.col_lower_ = np.asarray(model.col_lb, dtype=float)
    lp.col_upper_ = np.asarray(model.col_ub, dtype=float)
    lp.row_lower_ = np.asarray(model.row_lb, dtype=float)
    lp.row_upper_ = np.asarray(model.row_ub, dtype=float)
    lp.offset_ = float(model.obj_offset)
    lp.sense_ = highspy.ObjSense.kMaximize if model.sense == "max" else highspy.ObjSense.kMinimize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(float)
    is_mip = bool(np.any(model.integer))
    if is_mip:
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if f else highspy.HighsVarType.kContinuous
            for f in model.integer
        ]
    h.passModel(lp)
    if is_mip and warm_start is not None:
        s = highspy.HighsSolution()
        s.col_value = list(np.asarray(warm_start, dtype=float))
        s.value_valid = True
        h.setSolution(s)

    t0 = time.perf_counter()
    h.run()
    runtime = time.perf_counter() - t0
    ms = h.getModelStatus()
    status = {
        highspy.HighsModelStatus.kOptimal: "optimal",
        highspy.HighsModelStatus.kTimeLimit: "time_limit",
        highspy.HighsModelStatus.kInfeasible: "infeasible",
    }.get(ms, "other")
    info = h.getInfo()
    sol = h.getSolution()
    x = np.asarray(sol.col_value, dtype=float)
    if status in ("infeasible", "other") and not sol.value_valid:
        return Solution(status, np.nan, np.array([]), np.nan, runtime, "highs")
    obj = float(info.objective_function_value)
    if is_mip:
        bound = float(info.mip_dual_bound)
        return Solution(status, obj, x, bound, runtime, "highs",
                        extra={"nodes": int(info.mip_node_count)})
    # HiGHS reports duals with the sign of d(obj)/d(rhs) for the stated sense already.
    duals = np.asarray(sol.row_dual, dtype=float)
    rc = np.asarray(sol.col_dual, dtype=float)
    return Solution(status, obj, x, obj, runtime, "highs", row_duals=duals, reduced_costs=rc)


# ------------------------------------------------------------------------- helpers
def build(
    c, rows: sp.spmatrix | np.ndarray, row_lb, row_ub,
    integer=True, col_lb=0.0, col_ub=1.0, sense="max", row_names=None, obj_offset=0.0,
) -> LinearModel:
    """Convenience constructor; ``rows`` is a sparse matrix."""
    c = np.asarray(c, dtype=float)
    n = len(c)
    A = sp.csr_matrix(rows)
    return LinearModel(
        c=c, A=A,
        row_lb=np.broadcast_to(np.asarray(row_lb, float), (A.shape[0],)).copy(),
        row_ub=np.broadcast_to(np.asarray(row_ub, float), (A.shape[0],)).copy(),
        col_lb=np.broadcast_to(np.asarray(col_lb, float), (n,)).copy(),
        col_ub=np.broadcast_to(np.asarray(col_ub, float), (n,)).copy(),
        integer=np.broadcast_to(np.asarray(integer, bool), (n,)).copy(),
        sense=sense, obj_offset=obj_offset, row_names=row_names,
    )
