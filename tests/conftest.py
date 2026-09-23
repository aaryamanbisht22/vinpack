from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def pytest_collection_modifyitems(config, items):
    if not (RAW / "mknapcb1.txt").exists():
        skip = pytest.mark.skip(reason="benchmark data missing: run `make data`")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def raw() -> Path:
    return RAW


@pytest.fixture(scope="session")
def scenario():
    from vinpack.pipeline.adapter import load_default

    return load_default()
