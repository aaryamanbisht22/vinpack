"""Parser for D. Pisinger's 0-1 knapsack instances ("Where are the hard knapsack problems?", 2005).

Each ``knapPI_<type>_<n>_<R>.csv`` file holds many instances, each in the block::

    knapPI_1_50_1000_1
    n 50
    c 995
    z 8373
    time 0.00
    1,94,485,0          <- item id, profit, weight, x in the optimal solution
    ...
    -----
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vinpack.io.instances import KnapsackInstance


def read_pisinger(path: Path, limit: int | None = None) -> list[KnapsackInstance]:
    out: list[KnapsackInstance] = []
    lines = iter(Path(path).read_text().splitlines())
    for line in lines:
        name = line.strip()
        if not name or name.startswith("-"):
            continue
        n = int(next(lines).split()[1])
        c = int(next(lines).split()[1])
        z = int(next(lines).split()[1])
        next(lines)  # time
        rows = np.array([next(lines).split(",") for _ in range(n)], dtype=np.int64)
        out.append(
            KnapsackInstance(
                name=name,
                profits=rows[:, 1].copy(),
                weights=rows[:, 2].copy(),
                capacity=c,
                optimum=z,
                optimal_x=rows[:, 3].astype(np.int8),
            )
        )
        if limit and len(out) >= limit:
            break
    return out
