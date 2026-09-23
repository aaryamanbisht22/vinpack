# All targets set PYTHONPATH=src so they work even where the editable-install .pth
# is ignored (Python >= 3.12 skips .pth files carrying the macOS "hidden" flag).
export PYTHONPATH := src
RUN := uv run --no-sync

.PHONY: setup data test test-all lint bench bench-quick simulate pareto app docs publish demo-db clean

setup:            ## install deps (incl. dev) and fetch benchmark data
	uv sync
	$(RUN) python data/fetch.py

data:
	$(RUN) python data/fetch.py

test:             ## fast tests (CI)
	$(RUN) pytest -q -m "not slow"

test-all:         ## includes slow benchmark-correctness tests
	$(RUN) pytest -q

lint:
	$(RUN) ruff check src tests benchmarks app

bench:            ## full per-layer benchmarks -> results/benchmarks.{csv,md}
	$(RUN) python -m vinpack.cli bench

bench-quick:
	$(RUN) python -m vinpack.cli bench --quick

simulate:         ## 14-day rolling re-solve stored in results/vinpack.duckdb
	$(RUN) python -m vinpack.cli simulate --days 14 --churn 0.5

pareto:
	$(RUN) python -m vinpack.cli pareto

app:
	$(RUN) streamlit run app/streamlit_app.py --server.headless true --server.port $(or $(PORT),8501)

docs:
	$(RUN) mkdocs build --strict

clean:
	rm -rf results/*.duckdb site .pytest_cache

publish:          ## regenerate docs/results.md from results/ and the DuckDB store
	$(RUN) python benchmarks/publish.py

demo-db:          ## rebuild the committed demo store the hosted dashboard starts from
	rm -f results/demo.duckdb
	$(RUN) python -m vinpack.cli simulate --days 14 --churn 0.5 --db results/demo.duckdb
	$(RUN) python -m vinpack.cli pareto --db results/demo.duckdb
