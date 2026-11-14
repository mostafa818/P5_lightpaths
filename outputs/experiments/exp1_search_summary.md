## Experiment 1: BFS vs UCS vs A*

| Graph | Algorithm | Expanded | Generated | Path cost (km) | Hops | Time (ms) |
|---|---|---|---|---|---|---|
| Abilene | BFS | 7.0 | 8.9 | 2308.9 | 2.50 | 0.004 ± 0.001 |
| Abilene | UCS | 7.0 | 9.1 | 2210.9 | 2.59 | 0.007 ± 0.002 |
| Abilene | A* | 4.0 | 6.8 | 2210.9 | 2.59 | 0.009 ± 0.003 |
| germany50 | BFS | 26.6 ± 1.1 | 33.6 ± 1.1 | 426.9 ± 14.6 | 4.09 ± 0.17 | 0.010 ± 0.001 |
| germany50 | UCS | 26.6 ± 0.8 | 36.7 ± 1.1 | 378.9 ± 14.0 | 4.54 ± 0.14 | 0.030 ± 0.001 |
| germany50 | A* | 7.6 ± 0.5 | 16.7 ± 0.6 | 378.9 ± 14.0 | 4.54 ± 0.14 | 0.020 ± 0.002 |

| Graph | Comparison | Pairs | Result |
|---|---|---|---|
| Abilene | A* vs UCS | 132 | A* fewer expanded in 86.4%, equal 13.6%, more 0.0%; mean ratio expanded A*/UCS = 0.620; cost equal in 100.0% |
| Abilene | BFS vs UCS | 132 | BFS path longer than optimal in 22.7% of pairs (mean excess 3.8% overall, 16.8% when longer); BFS has fewer hops than the UCS path in 9.1%, equal 90.9%, more 0.0% |
| germany50 | A* vs UCS | 500 | A* fewer expanded in 97.2%, equal 2.8%, more 0.0%; mean ratio expanded A*/UCS = 0.332; cost equal in 100.0% |
| germany50 | BFS vs UCS | 500 | BFS path longer than optimal in 56.0% of pairs (mean excess 11.1% overall, 19.8% when longer); BFS has fewer hops than the UCS path in 29.0%, equal 71.0%, more 0.0% |

Independent check: BFS hop count equals the networkx unweighted shortest path in all pairs (0 mismatches).
