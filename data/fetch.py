"""Download the benchmark data (thin wrapper so it works before the package is installed).

    uv run python data/fetch.py [--no-pisinger]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vinpack.io.fetch import main  # noqa: E402

if __name__ == "__main__":
    main()
