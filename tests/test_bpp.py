import pytest

from vinpack.io.orlib import read_binpack
from vinpack.solvers import bpp
from vinpack.solvers.arcflow import arcflow
from vinpack.solvers.colgen import column_generation


@pytest.mark.parametrize("f,k", [("binpack1", 6), ("binpack5", 6)])
def test_colgen_proves_optimum(raw, f, k):
    for inst in read_binpack(raw / f"{f}.txt")[:k]:
        r = column_generation(inst)
        assert bpp.is_feasible(inst, r.bins)
        assert r.proven_optimal, inst.name
        assert r.n_bins <= inst.best_known


def test_lower_bounds_are_valid(raw):
    for inst in read_binpack(raw / "binpack1.txt"):
        assert bpp.lower_bound_l1(inst.sizes, inst.capacity) <= bpp.lower_bound_l2(
            inst.sizes, inst.capacity) <= inst.best_known


def test_ffd_feasible(raw):
    inst = read_binpack(raw / "binpack2.txt")[0]
    r = bpp.ffd(inst)
    assert bpp.is_feasible(inst, r.bins) and r.n_bins >= inst.best_known


def test_arcflow_exact_on_triplet(raw):
    inst = read_binpack(raw / "binpack5.txt")[6]
    r = arcflow(inst)
    assert bpp.is_feasible(inst, r.bins)
    assert r.n_bins == inst.best_known == r.lower_bound
