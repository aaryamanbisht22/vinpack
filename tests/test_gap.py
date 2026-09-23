import pytest

from vinpack.io.orlib import read_gap
from vinpack.solvers import gap


@pytest.mark.parametrize("f", ["gap1", "gap4", "gap8", "gap12"])
def test_mip_matches_published_optimum(raw, f):
    for inst in read_gap(raw / f"{f}.txt"):
        r = gap.solve_mip(inst, time_limit=30)
        assert r.value == inst.optimum, inst.name
        assert gap.is_feasible(inst, r.assign)


def test_greedy_is_feasible_and_no_better_than_optimum(raw):
    for inst in read_gap(raw / "gap12.txt"):
        g = gap.greedy_regret(inst)
        if g.status != "infeasible":
            assert gap.is_feasible(inst, g.assign)
            assert g.value <= inst.optimum  # maximisation
