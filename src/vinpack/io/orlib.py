"""Parsers for J.E. Beasley's OR-Library files (mknapcb*, gap*, binpack*).

All OR-Library files are whitespace-separated token streams, so each parser
reads the whole file into a token iterator and consumes it field by field in
the order given in the library's format notes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from vinpack.io.instances import BinPackingInstance, GAPInstance, MKPInstance

# Published optima for gap1-12 (maximisation), from OR-Library's gapinfo page.
GAP_OPTIMA_MAX: dict[str, list[int]] = {
    "gap1": [336, 327, 339, 341, 326],
    "gap2": [434, 436, 420, 419, 428],
    "gap3": [580, 564, 573, 570, 564],
    "gap4": [656, 644, 673, 647, 664],
    "gap5": [563, 558, 564, 568, 559],
    "gap6": [761, 759, 758, 752, 747],
    "gap7": [942, 949, 968, 945, 951],
    "gap8": [1133, 1134, 1141, 1117, 1127],
    "gap9": [709, 717, 712, 723, 706],
    "gap10": [958, 963, 960, 947, 947],
    "gap11": [1139, 1178, 1195, 1171, 1171],
    "gap12": [1451, 1449, 1433, 1447, 1446],
}

# Falkenauer 'u' instances whose file value is not a proven optimum (OR-Library notes).
_BPP_NOT_PROVEN = {"u120_08", "u120_19", "u250_07", "u250_12", "u250_13"}

# mknapcbK holds 30 problems of size (m, n); this is how Chu & Beasley name them.
_MKNAPCB_SHAPES = {
    1: (5, 100), 2: (5, 250), 3: (5, 500),
    4: (10, 100), 5: (10, 250), 6: (10, 500),
    7: (30, 100), 8: (30, 250), 9: (30, 500),
}


def _tokens(path: Path) -> Iterator[str]:
    return iter(Path(path).read_text().split())


def _ints(tok: Iterator[str], k: int) -> np.ndarray:
    return np.fromiter((int(float(next(tok))) for _ in range(k)), dtype=np.int64, count=k)


def read_mkcbres(path: Path) -> tuple[dict[str, int], dict[str, float]]:
    """Best-known values and LP bounds from ``mkcbres.txt``, keyed like '5.100-00'."""
    best: dict[str, int] = {}
    lp: dict[str, float] = {}
    target = best
    for line in Path(path).read_text().splitlines():
        if "LP optimal" in line:
            target = lp
        m = re.match(r"^\s*(\d+\.\d+-\d+)\s+([\d.eE+-]+)\s*$", line)
        if m:
            key, val = m.groups()
            target[key] = float(val) if target is lp else int(float(val))
    return best, lp


def read_mknap(path: Path, results: Path | None = None) -> list[MKPInstance]:
    """Parse an mknap1 / mknapcbK file. ``results`` is the optional mkcbres.txt."""
    path = Path(path)
    tok = _tokens(path)
    k = int(next(tok))
    best, lp = read_mkcbres(results) if results and Path(results).exists() else ({}, {})
    file_no = re.search(r"mknapcb(\d)", path.name)
    out = []
    for idx in range(k):
        n, m, opt = int(next(tok)), int(next(tok)), int(float(next(tok)))
        profits = _ints(tok, n)
        resources = np.vstack([_ints(tok, n) for _ in range(m)])
        caps = _ints(tok, m)
        if file_no:
            key = f"{m}.{n}-{idx:02d}"
            name = f"mknapcb{file_no.group(1)}:{key}"
        else:
            key, name = "", f"{path.stem}:{idx}"
        out.append(
            MKPInstance(
                name=name,
                profits=profits,
                resources=resources,
                capacities=caps,
                best_known=best.get(key) or (opt or None),
                lp_bound=lp.get(key),
            )
        )
    return out


def read_gap(path: Path) -> list[GAPInstance]:
    """Parse gap1-12 (maximisation optima known) or gapa-d (minimisation)."""
    path = Path(path)
    tok = _tokens(path)
    p = int(next(tok))
    stem = path.stem
    is_letter = stem in {"gapa", "gapb", "gapc", "gapd"}
    optima = GAP_OPTIMA_MAX.get(stem, [])
    out = []
    for idx in range(p):
        m, n = int(next(tok)), int(next(tok))
        costs = np.vstack([_ints(tok, n) for _ in range(m)])
        res = np.vstack([_ints(tok, n) for _ in range(m)])
        caps = _ints(tok, m)
        out.append(
            GAPInstance(
                name=f"{stem}:{m}x{n}-{idx + 1}",
                costs=costs,
                resources=res,
                capacities=caps,
                sense="min" if is_letter else "max",
                optimum=optima[idx] if idx < len(optima) else None,
            )
        )
    return out


def read_binpack(path: Path) -> list[BinPackingInstance]:
    """Parse Falkenauer's binpack1-8 files."""
    tok = _tokens(path)
    p = int(next(tok))
    out = []
    for _ in range(p):
        name = next(tok)
        cap_s, n, best = next(tok), int(next(tok)), int(next(tok))
        raw = [next(tok) for _ in range(n)]
        # Triplet sizes carry one decimal (e.g. 26.3). Scale to exact integers so the
        # DP-based solvers stay exact; the scale is kept in meta for reporting.
        decimals = max((len(s.split(".")[1].rstrip("0")) for s in raw + [cap_s] if "."
                        in s), default=0)
        scale = 10**decimals
        sizes = np.array([round(float(s) * scale) for s in raw], dtype=np.int64)
        cap = round(float(cap_s) * scale)
        triplet = name.startswith("t")
        out.append(
            BinPackingInstance(
                name=name,
                sizes=sizes,
                capacity=cap,
                best_known=best,
                proven_optimal=triplet or name not in _BPP_NOT_PROVEN,
                meta={"family": "falkenauer_t" if triplet else "falkenauer_u", "scale": scale},
            )
        )
    return out


def mknapcb_shape(file_no: int) -> tuple[int, int]:
    return _MKNAPCB_SHAPES[file_no]
