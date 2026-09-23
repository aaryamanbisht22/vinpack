"""Download every academic benchmark set vinpack uses into data/raw/.

    vinpack fetch                     # OR-Library + Pisinger small-coefficient subset
    vinpack fetch --no-pisinger

Raw files are gitignored; data/MANIFEST.sha256 is committed so a fresh clone
can verify it downloaded byte-identical instances.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "data"
RAW = ROOT / "raw"
ORLIB = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"
PISINGER = "https://hjemmesider.diku.dk/~pisinger/smallcoeff_pisinger.tgz"

ORLIB_FILES = (
    [f"mknapcb{i}.txt" for i in range(1, 10)]
    + ["mkcbres.txt"]
    + [f"gap{i}.txt" for i in range(1, 13)]
    + [f"gap{c}.txt" for c in "abcd"]
    + [f"binpack{i}.txt" for i in range(1, 9)]
)
# Pisinger instance types: 1 uncorrelated, 2 weakly corr., 3 strongly corr., 4 inverse strongly corr.
PISINGER_KEEP = {
    f"knapPI_{t}_{n}_1000.csv" for t in (1, 2, 3, 4) for n in (50, 100, 200, 500, 1000, 2000)
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vinpack-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def fetch_orlib(names: list[str] | None = None) -> None:
    """Download OR-Library files (all of them, or just ``names``) that are not present yet."""
    for name in names or ORLIB_FILES:
        dest = RAW / name
        if dest.exists():
            continue
        print(f"  OR-Library  {name}")
        dest.write_bytes(_get(ORLIB + name))


def fetch_pisinger(archive: Path | None = None) -> None:
    out = RAW / "pisinger"
    if out.exists() and len(list(out.glob("*.csv"))) == len(PISINGER_KEEP):
        return
    out.mkdir(exist_ok=True)
    print("  Pisinger    smallcoeff_pisinger.tgz (~150 MB, slow server)")
    blob = archive.read_bytes() if archive and archive.exists() else _get(PISINGER)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for member in tf.getmembers():
            base = Path(member.name).name
            if base in PISINGER_KEEP:
                (out / base).write_bytes(tf.extractfile(member).read())


def write_manifest() -> None:
    lines = []
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.suffix in {".txt", ".csv"}:
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(RAW)}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def verify_manifest() -> list[str]:
    bad = []
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        return bad
    for line in manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = RAW / rel
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pisinger", action="store_true")
    ap.add_argument("--pisinger-archive", type=Path, help="use an already-downloaded tgz")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    print("Fetching benchmark data into", RAW)
    fetch_orlib()
    if not args.no_pisinger:
        fetch_pisinger(args.pisinger_archive)
    if args.write_manifest:
        write_manifest()
    bad = verify_manifest()
    if bad:
        raise SystemExit(f"checksum mismatch: {bad}")
    print("Done.")


if __name__ == "__main__":
    main()
