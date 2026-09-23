| Layer | Instances | Method | n | = reference | proven opt. | mean gap | worst gap | always feasible | mean s |
|---|---|---|---:|---:|---:|---:|---:|:---:|---:|
| L0 knapsack | Pisinger uncorrelated | DP (numba) | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.01 |
| L0 knapsack | Pisinger uncorrelated | branch & bound | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger weakly corr. | DP (numba) | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger weakly corr. | branch & bound | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger strongly corr. | DP (numba) | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger strongly corr. | branch & bound | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger inv. strongly corr. | DP (numba) | 15 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger inv. strongly corr. | branch & bound | 15 | 100% | 93% | 0.000% | 0.000% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=100 | dual-ratio greedy + 1-swap | 3 | 0% | 0% | 0.546% | 0.726% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=100 | MIP (highs, 5s) | 3 | 100% | 100% | 0.000% | 0.000% | yes | 1.65 |
| L2 matching (GAP) | OR-Library gap1-12 | MIP (highs, 5s) | 6 | 100% | 100% | 0.000% | 0.000% | yes | 0.10 |
| L2 matching (GAP) | OR-Library gap1-12 | regret greedy + local search | 6 | 0% | 0% | 1.158% | 1.488% | no | 0.00 |
| L2 matching (GAP) | OR-Library gapa (type A) | MIP (highs, 5s) | 2 | 100% | 100% | 0.000% | 0.000% | yes | 0.01 |
| L2 matching (GAP) | OR-Library gapa (type A) | regret greedy + local search | 2 | 100% | 0% | 0.000% | 0.000% | yes | 0.03 |
| L3 loading (BPP) | Falkenauer uniform n=120 | FFD | 5 | 40% | 40% | 1.260% | 2.174% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer uniform n=120 | BFD | 5 | 40% | 40% | 1.260% | 2.174% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer uniform n=120 | column generation (+dive, +arc-flow) | 5 | 100% | 100% | 0.000% | 0.000% | yes | 0.15 |
| L3 loading (BPP) | Falkenauer uniform n=120 | compact assignment MIP (10s) | 5 | 40% | 40% | 1.260% | 2.174% | yes | 6.06 |
| L3 loading (BPP) | Falkenauer triplet n=60 | FFD | 5 | 0% | 0% | 16.000% | 20.000% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=60 | BFD | 5 | 0% | 0% | 16.000% | 20.000% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=60 | column generation (+dive, +arc-flow) | 5 | 100% | 100% | 0.000% | 0.000% | yes | 0.18 |
| L3 loading (BPP) | Falkenauer triplet n=60 | compact assignment MIP (10s) | 5 | 0% | 0% | 5.000% | 5.000% | yes | 10.00 |
