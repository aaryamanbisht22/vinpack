import pytest

from vinpack.io.orlib import read_gap
from vinpack.solvers import gap
from vinpack.solvers.lagrangian import gap_lagrangian


@pytest.mark.parametrize("f", ["gap1", "gap4", "gap8", "gap12"])
def test_mip_matches_published_optimum(raw, f):
    for inst in read_gap(raw / f"{f}.txt"):
        r = gap.solve_mip(inst, time_limit=30)
        assert r.value == inst.optimum, inst.name
        assert gap.is_feasible(inst, r.assign)


@pytest.mark.parametrize("f", ["gap1", "gap12"])
def test_lagrangian_bound_valid_and_solution_feasible(raw, f):
    for inst in read_gap(raw / f"{f}.txt"):
        L = gap_lagrangian(inst)
        assert L.bound >= inst.optimum  # maximisation: an upper bound
        if L.x.min() >= 0:
            assert gap.is_feasible(inst, L.x)
            assert L.value <= inst.optimum


def test_min_instances_bound_below_solution(raw):
    inst = read_gap(raw / "gapa.txt")[0]
    r = gap.solve_mip(inst, time_limit=30)
    L = gap_lagrangian(inst)
    assert L.bound <= r.value <= L.value
