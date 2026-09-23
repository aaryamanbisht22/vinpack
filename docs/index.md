# vinpack

**Allocate scarce supply to orders, match them to delivery centers, and load them onto carriers,
then re-solve all of it every day without reshuffling customers.** Every layer is a
knapsack-family model, every number comes from a published academic benchmark, and every
explanation is checked against solver output.

* [How it works](how-it-works.md): the three layers and the daily stability term
* [Formulations](formulations.md): the models, the relaxations, column generation and arc-flow
* [Results](results.md): per-layer benchmarks against published optima, and the value-vs-stability frontier

```bash
git clone https://github.com/aaryamanbisht22/vinpack && cd vinpack
make setup        # uv sync + download the benchmark data (checksummed)
make simulate     # 14-day rolling re-solve -> results/vinpack.duckdb
make app          # dashboard: daily plan, diffs, ask-why agent, frontier, benchmarks
```
