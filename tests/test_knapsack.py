from pathlib import Path

import pytest

from vinpack.io.pisinger import read_pisinger
from vinpack.solvers.kp_dp import kp_bb, kp_dp

HAS_PISINGER = (Path(__file__).resolve().parents[1] / "data" / "raw" / "pisinger").exists()
needs_pisinger = pytest.mark.skipif(not HAS_PISINGER, reason="Pisinger instances not downloaded")


@needs_pisinger
@pytest.mark.parametrize("kind", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [50, 100, 200])
def test_dp_matches_pisinger_optimum(raw, kind, n):
    for inst in read_pisinger(raw / "pisinger" / f"knapPI_{kind}_{n}_1000.csv", limit=10):
        value, x = kp_dp(inst.profits, inst.weights, inst.capacity)
        assert value == inst.optimum
        assert x @ inst.weights <= inst.capacity
        assert x @ inst.profits == value


@needs_pisinger
@pytest.mark.parametrize("kind", [1, 2])
def test_branch_and_bound_matches_pisinger(raw, kind):
    for inst in read_pisinger(raw / "pisinger" / f"knapPI_{kind}_200_1000.csv", limit=10):
        value, x, _, proven = kp_bb(inst.profits, inst.weights, inst.capacity)
        assert proven and value == inst.optimum
        assert x @ inst.weights <= inst.capacity


def test_dp_float_profits_for_pricing():
    # column-generation pricing uses fractional duals as profits
    value, x = kp_dp([0.4, 0.35, 0.3], [60, 50, 40], 100)
    assert abs(value - 0.7) < 1e-12 and list(x) == [1, 0, 1]
