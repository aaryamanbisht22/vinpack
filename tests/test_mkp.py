from dataclasses import replace

import numpy as np

from vinpack.io.orlib import read_mknap
from vinpack.solvers import mkp
from vinpack.solvers.lagrangian import mkp_lagrangian


def _insts(raw):
    return read_mknap(raw / "mknapcb1.txt", raw / "mkcbres.txt")


def test_mip_reaches_best_known(raw):
    for idx in (0, 15):
        inst = _insts(raw)[idx]
        r = mkp.solve_mip(inst, time_limit=30)
        assert r.status == "optimal"
        assert r.value == inst.best_known
        assert mkp.is_feasible(inst, r.x)


def test_lp_matches_published_lp_bound(raw):
    inst = _insts(raw)[0]
    assert abs(mkp.solve_lp(inst).value - inst.lp_bound) < 1e-4


def test_duals_are_marginal_values(raw):
    inst = _insts(raw)[15]
    lp = mkp.solve_lp(inst)
    i = int(np.argmax(lp.duals))
    caps = inst.capacities.copy()
    caps[i] += 1
    lp2 = mkp.solve_lp(replace(inst, capacities=caps))
    assert abs((lp2.value - lp.value) - lp.duals[i]) < 1e-6


def test_heuristics_feasible_and_bounds_valid(raw):
    inst = _insts(raw)[0]
    lp = mkp.solve_lp(inst)
    g = mkp.greedy_dual(inst, lp.duals)
    L = mkp_lagrangian(inst, lp_duals=lp.duals)
    assert mkp.is_feasible(inst, g.x) and mkp.is_feasible(inst, L.x)
    assert g.value <= inst.best_known and L.value <= inst.best_known
    # the KP-subproblem Lagrangian bound is valid and at least as tight as the LP bound
    assert inst.best_known <= L.bound <= np.floor(lp.value)


def test_stability_offset_keeps_objective_consistent(raw):
    inst = _insts(raw)[0]
    base = mkp.solve_mip(inst, time_limit=30)
    prev = base.x.astype(float)
    stab = mkp.Stability(prev, weight=1e4)
    r = mkp.solve_mip(inst, time_limit=30, stability=stab)
    assert np.array_equal(r.x, base.x)  # huge churn weight keeps yesterday's optimum
    assert abs(r.extra["objective_with_churn"] - r.value) < 1e-6
