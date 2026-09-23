# Benchmark data sources

Raw files are not committed. `make data` downloads them and `data/MANIFEST.sha256` checks them.

| Set | Used for | Source | Reference |
|---|---|---|---|
| `mknapcb1-9`, `mkcbres` | L1 allocation (MKP) | OR-Library, J.E. Beasley | P.C. Chu and J.E. Beasley, "A genetic algorithm for the multidimensional knapsack problem", *Journal of Heuristics* 4 (1998) 63-86 |
| `gap1-12` | L2 matching (GAP), published optima | OR-Library | I.H. Osman, *OR Spektrum* 17 (1995) 211-225; D. Cattrysse, M. Salomon, L.N. Van Wassenhove, *EJOR* 72 (1994) 167-174 |
| `gapa-d` | L2 matching (GAP), large instances | OR-Library | P.C. Chu and J.E. Beasley, "A genetic algorithm for the generalised assignment problem", *Computers & OR* 24 (1997) 17-23 |
| `binpack1-8` | L3 loading (BPP) | OR-Library | E. Falkenauer, "A hybrid grouping genetic algorithm for bin packing", *Journal of Heuristics* 2 (1996) 5-30 |
| `knapPI_{1-4}_*_1000` | L0 knapsack kernel | D. Pisinger | D. Pisinger, "Where are the hard knapsack problems?", *Computers & OR* 32 (2005) 2271-2284 |

Methods: P.C. Gilmore and R.E. Gomory, *Operations Research* 9 (1961); J.M. Valério de Carvalho, *Annals of OR* 86 (1999);
S. Martello and P. Toth, *Knapsack Problems* (Wiley, 1990).
