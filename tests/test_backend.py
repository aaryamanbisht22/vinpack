import numpy as np
import pytest

from vinpack.io.orlib import read_gap, read_mknap
from vinpack.solvers import gap, mkp
from vinpack.solvers.backend import available_backends

needs_gurobi = pytest.mark.skipif("gurobi" not in available_backends(),
                                  reason="gurobipy / license not available")


@needs_gurobi
def test_backends_agree_on_mkp(raw):
    inst = read_mknap(raw / "mknapcb1.txt", raw / "mkcbres.txt")[15]
    h = mkp.solve_mip(inst, "highs", 30)
    g = mkp.solve_mip(inst, "gurobi", 30)
    assert h.value == g.value == inst.best_known
    lh, lg = mkp.solve_lp(inst, "highs"), mkp.solve_lp(inst, "gurobi")
    assert abs(lh.value - lg.value) < 1e-6
    assert np.allclose(lh.duals, lg.duals, atol=1e-6)


@needs_gurobi
def test_backends_agree_on_gap(raw):
    inst = read_gap(raw / "gap12.txt")[0]
    assert gap.solve_mip(inst, "highs").value == gap.solve_mip(inst, "gurobi").value == inst.optimum
