# Results

All numbers are produced by `make bench` and `make pareto` on an Apple M4 Pro with HiGHS (open source). Nothing on this page is typed in by hand.

## Per-layer benchmarks on unmodified academic instances

*= reference*: matches or beats the published optimum or best-known value. *proven opt.*: the method itself proved optimality (bound = value). *gap*: relative distance to the reference (positive = worse). For gapa-d the reference is this project's own proven MIP optimum when one was found.

| Layer | Instances | Method | n | = reference | proven opt. | mean gap | worst gap | always feasible | mean s |
|---|---|---|---:|---:|---:|---:|---:|:---:|---:|
| L0 knapsack | Pisinger uncorrelated | DP (numba) | 120 | 100% | 100% | 0.000% | 0.000% | yes | 0.03 |
| L0 knapsack | Pisinger uncorrelated | branch & bound | 120 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger weakly corr. | DP (numba) | 120 | 100% | 100% | 0.000% | 0.000% | yes | 0.02 |
| L0 knapsack | Pisinger weakly corr. | branch & bound | 120 | 100% | 100% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger strongly corr. | DP (numba) | 120 | 100% | 100% | 0.000% | 0.000% | yes | 0.03 |
| L0 knapsack | Pisinger strongly corr. | branch & bound | 120 | 100% | 43% | 0.000% | 0.000% | yes | 0.00 |
| L0 knapsack | Pisinger inv. strongly corr. | DP (numba) | 120 | 100% | 100% | 0.000% | 0.000% | yes | 0.03 |
| L0 knapsack | Pisinger inv. strongly corr. | branch & bound | 120 | 100% | 42% | 0.000% | 0.000% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=100 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.519% | 0.777% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=100 | MIP (highs, 10s) | 9 | 89% | 89% | 0.002% | 0.019% | yes | 3.35 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=250 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.279% | 0.782% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=250 | MIP (highs, 10s) | 9 | 0% | 0% | 0.043% | 0.140% | yes | 10.00 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=500 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.131% | 0.383% | yes | 0.01 |
| L1 allocation (MKP) | Chu-Beasley m=5 n=500 | MIP (highs, 10s) | 9 | 33% | 0% | 0.007% | 0.030% | yes | 10.00 |
| L1 allocation (MKP) | Chu-Beasley m=10 n=100 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.961% | 2.201% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=10 n=100 | MIP (highs, 10s) | 9 | 56% | 11% | 0.069% | 0.224% | yes | 9.13 |
| L1 allocation (MKP) | Chu-Beasley m=10 n=250 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.402% | 1.012% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=10 n=250 | MIP (highs, 10s) | 9 | 22% | 0% | 0.044% | 0.159% | yes | 10.00 |
| L1 allocation (MKP) | Chu-Beasley m=10 n=500 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.203% | 0.499% | yes | 0.01 |
| L1 allocation (MKP) | Chu-Beasley m=10 n=500 | MIP (highs, 10s) | 9 | 0% | 0% | 0.036% | 0.072% | yes | 10.00 |
| L1 allocation (MKP) | Chu-Beasley m=30 n=100 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 1.918% | 5.387% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=30 n=100 | MIP (highs, 10s) | 9 | 22% | 0% | 0.128% | 0.571% | yes | 10.00 |
| L1 allocation (MKP) | Chu-Beasley m=30 n=250 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.683% | 1.164% | yes | 0.00 |
| L1 allocation (MKP) | Chu-Beasley m=30 n=250 | MIP (highs, 10s) | 9 | 44% | 0% | 0.023% | 0.173% | yes | 10.00 |
| L1 allocation (MKP) | Chu-Beasley m=30 n=500 | dual-ratio greedy + 1-swap | 9 | 0% | 0% | 0.533% | 1.426% | yes | 0.01 |
| L1 allocation (MKP) | Chu-Beasley m=30 n=500 | MIP (highs, 10s) | 9 | 0% | 0% | 0.064% | 0.188% | yes | 10.00 |
| L2 matching (GAP) | OR-Library gap1-12 | MIP (highs, 20s) | 60 | 100% | 100% | 0.000% | 0.000% | yes | 0.14 |
| L2 matching (GAP) | OR-Library gap1-12 | regret greedy + local search | 60 | 0% | 0% | 1.616% | 3.667% | no | 0.00 |
| L2 matching (GAP) | OR-Library gapa (type A) | MIP (highs, 20s) | 6 | 100% | 100% | 0.000% | 0.000% | yes | 0.02 |
| L2 matching (GAP) | OR-Library gapa (type A) | regret greedy + local search | 6 | 83% | 0% | 0.007% | 0.043% | yes | 0.03 |
| L2 matching (GAP) | OR-Library gapb (type B) | MIP (highs, 20s) | 6 | 100% | 100% | 0.000% | 0.000% | yes | 1.35 |
| L2 matching (GAP) | OR-Library gapb (type B) | regret greedy + local search | 6 | 0% | 0% | 3.039% | 4.019% | no | 0.03 |
| L2 matching (GAP) | OR-Library gapc (type C) | MIP (highs, 20s) | 6 | 100% | 100% | 0.000% | 0.000% | yes | 2.94 |
| L2 matching (GAP) | OR-Library gapc (type C) | regret greedy + local search | 6 | 0% | 0% | – | – | no | 0.03 |
| L2 matching (GAP) | OR-Library gapd (type D) | MIP (highs, 20s) | 6 | 0% | 0% | – | – | yes | 20.00 |
| L2 matching (GAP) | OR-Library gapd (type D) | regret greedy + local search | 6 | 0% | 0% | – | – | no | 0.04 |
| L3 loading (BPP) | Falkenauer uniform n=120 | FFD | 20 | 40% | 30% | 1.243% | 2.174% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer uniform n=120 | BFD | 20 | 40% | 30% | 1.243% | 2.174% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer uniform n=120 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | -0.198% | 0.000% | yes | 0.20 |
| L3 loading (BPP) | Falkenauer uniform n=120 | compact assignment MIP (10s) | 3 | 33% | 33% | 1.419% | 2.174% | yes | 6.71 |
| L3 loading (BPP) | Falkenauer uniform n=250 | FFD | 20 | 0% | 0% | 1.378% | 2.970% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer uniform n=250 | BFD | 20 | 0% | 0% | 1.378% | 2.970% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer uniform n=250 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | -0.095% | 0.000% | yes | 0.34 |
| L3 loading (BPP) | Falkenauer uniform n=500 | FFD | 20 | 0% | 0% | 1.343% | 1.531% | yes | 0.01 |
| L3 loading (BPP) | Falkenauer uniform n=500 | BFD | 20 | 0% | 0% | 1.343% | 1.531% | yes | 0.01 |
| L3 loading (BPP) | Falkenauer uniform n=500 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | 0.000% | 0.000% | yes | 0.60 |
| L3 loading (BPP) | Falkenauer uniform n=1000 | FFD | 20 | 0% | 0% | 1.211% | 1.763% | yes | 0.02 |
| L3 loading (BPP) | Falkenauer uniform n=1000 | BFD | 20 | 0% | 0% | 1.211% | 1.763% | yes | 0.02 |
| L3 loading (BPP) | Falkenauer uniform n=1000 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | 0.000% | 0.000% | yes | 0.85 |
| L3 loading (BPP) | Falkenauer triplet n=60 | FFD | 20 | 0% | 0% | 16.000% | 20.000% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=60 | BFD | 20 | 0% | 0% | 16.000% | 20.000% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=60 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | 0.000% | 0.000% | yes | 0.25 |
| L3 loading (BPP) | Falkenauer triplet n=60 | compact assignment MIP (10s) | 3 | 0% | 0% | 5.000% | 5.000% | yes | 10.00 |
| L3 loading (BPP) | Falkenauer triplet n=120 | FFD | 20 | 0% | 0% | 14.500% | 17.500% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=120 | BFD | 20 | 0% | 0% | 14.500% | 17.500% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=120 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | 0.000% | 0.000% | yes | 2.64 |
| L3 loading (BPP) | Falkenauer triplet n=120 | compact assignment MIP (10s) | 3 | 0% | 0% | 5.000% | 5.000% | yes | 10.01 |
| L3 loading (BPP) | Falkenauer triplet n=249 | FFD | 20 | 0% | 0% | 14.458% | 15.663% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=249 | BFD | 20 | 0% | 0% | 14.458% | 15.663% | yes | 0.00 |
| L3 loading (BPP) | Falkenauer triplet n=249 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | 0.000% | 0.000% | yes | 12.56 |
| L3 loading (BPP) | Falkenauer triplet n=501 | FFD | 20 | 0% | 0% | 13.802% | 14.371% | yes | 0.01 |
| L3 loading (BPP) | Falkenauer triplet n=501 | BFD | 20 | 0% | 0% | 13.802% | 14.371% | yes | 0.01 |
| L3 loading (BPP) | Falkenauer triplet n=501 | column generation (+dive, +arc-flow) | 20 | 100% | 100% | 0.000% | 0.000% | yes | 30.75 |


*gapd*: no published optimum is stored and the 20 s MIP does not prove one, so those rows show no gap to a reference. Their gap to the MIP's own proven bound is in `results/benchmarks.csv` (`bound` column): 0.05-1.3%.


## Daily re-solve: value vs. plan stability

14-day rolling simulation on the composed scenario (seed 7). λ is the churn penalty as a multiple of the mean order value (L1) and the mean lane cost (L2).

| churn weight λ | plan value (14 days) | vs λ=0 | orders re-planned | vs λ=0 | DC reassignments | broken promises | carriers | days with proven-optimal loading |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,622,360 | +0.0% | 171 | +0% | 95 | 1 | 930 | 100% |
| 0.1 | 1,613,792 | -0.5% | 98 | -43% | 46 | 1 | 925 | 100% |
| 0.25 | 1,599,488 | -1.4% | 54 | -68% | 26 | 2 | 908 | 100% |
| 0.5 | 1,593,238 | -1.8% | 41 | -76% | 11 | 2 | 891 | 100% |
| 1 | 1,551,865 | -4.3% | 10 | -94% | 12 | 1 | 877 | 100% |
| 2 | 1,533,388 | -5.5% | 1 | -99% | 3 | 2 | 867 | 100% |
