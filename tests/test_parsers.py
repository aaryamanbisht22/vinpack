import numpy as np

from vinpack.io.orlib import read_binpack, read_gap, read_mkcbres, read_mknap
from vinpack.io.pisinger import read_pisinger


def test_mknap_shapes_and_references(raw):
    insts = read_mknap(raw / "mknapcb1.txt", raw / "mkcbres.txt")
    assert len(insts) == 30
    first = insts[0]
    assert (first.m, first.n) == (5, 100)
    assert first.best_known == 24381
    assert abs(first.lp_bound - 24585.902722) < 1e-6


def test_mkcbres_has_both_tables(raw):
    best, lp = read_mkcbres(raw / "mkcbres.txt")
    assert len(best) == len(lp) == 270
    assert all(lp[k] >= best[k] for k in best)  # LP bound never below a feasible value


def test_gap_optima_attached(raw):
    g = read_gap(raw / "gap1.txt")
    assert [i.optimum for i in g] == [336, 327, 339, 341, 326]
    assert g[0].sense == "max" and (g[0].m, g[0].n) == (5, 15)
    a = read_gap(raw / "gapa.txt")
    assert a[0].sense == "min" and a[0].optimum is None


def test_binpack_triplet_sizes_are_exact(raw):
    t = read_binpack(raw / "binpack5.txt")[0]
    # triplets fill every bin exactly: total size == n/3 * capacity after scaling
    assert t.meta["scale"] == 10
    assert t.sizes.sum() == (t.n // 3) * t.capacity
    u = read_binpack(raw / "binpack1.txt")[0]
    assert u.capacity == 150 and u.n == 120 and u.best_known == 48


def test_pisinger_solution_matches_value(raw):
    import pytest

    if not (raw / "pisinger").exists():
        pytest.skip("Pisinger instances not downloaded")
    for inst in read_pisinger(raw / "pisinger" / "knapPI_3_100_1000.csv", limit=5):
        assert inst.optimal_x @ inst.profits == inst.optimum
        assert inst.optimal_x @ inst.weights <= inst.capacity
        assert np.all(inst.weights > 0)
