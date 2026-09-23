"""Typed containers for every benchmark instance the pipeline consumes.

Each layer of the pipeline has exactly one instance type. Parsers in this
package turn the academic file formats into these objects; solvers only ever
see these objects, never raw files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class KnapsackInstance:
    """0-1 knapsack: max p.x  s.t.  w.x <= capacity."""

    name: str
    profits: np.ndarray  # (n,) int64
    weights: np.ndarray  # (n,) int64
    capacity: int
    optimum: int | None = None  # published optimal value, if known
    optimal_x: np.ndarray | None = None  # published optimal solution, if known

    @property
    def n(self) -> int:
        return len(self.profits)


@dataclass(frozen=True)
class MKPInstance:
    """Multidimensional knapsack: max p.x  s.t.  R x <= b,  x binary."""

    name: str
    profits: np.ndarray  # (n,)
    resources: np.ndarray  # (m, n)
    capacities: np.ndarray  # (m,)
    best_known: int | None = None
    lp_bound: float | None = None

    @property
    def n(self) -> int:
        return self.resources.shape[1]

    @property
    def m(self) -> int:
        return self.resources.shape[0]


@dataclass(frozen=True)
class GAPInstance:
    """Generalized assignment: every job goes to exactly one agent, agents have capacity.

    ``sense`` records how the published optimum is defined: OR-Library gap1-12
    optima are reported for maximisation, gapa-d for minimisation.
    """

    name: str
    costs: np.ndarray  # (m, n)  value/cost of job j on agent i
    resources: np.ndarray  # (m, n)
    capacities: np.ndarray  # (m,)
    sense: str = "max"  # "max" or "min"
    optimum: int | None = None

    @property
    def m(self) -> int:
        return self.costs.shape[0]

    @property
    def n(self) -> int:
        return self.costs.shape[1]


@dataclass(frozen=True)
class BinPackingInstance:
    """One-dimensional bin packing: fewest bins of size ``capacity`` holding all items."""

    name: str
    sizes: np.ndarray  # (n,)
    capacity: int
    best_known: int | None = None
    proven_optimal: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sizes)
