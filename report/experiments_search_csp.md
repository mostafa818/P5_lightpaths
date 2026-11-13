# Experiments 1 and 2 (Member B): results and reading

Reproduce: `python -m eval.run_experiments exp1 exp2` (Experiment 1 takes seconds, Experiment 2 about 14 minutes because
plain backtracking runs to the 1,000,000-assignment cap on the hardest instances). Raw rows are in
`outputs/experiments/exp1_search_raw.csv` and `exp2_csp_raw.csv`, the tables in `*_summary.md`, the plots in
`exp1_search.png` and `exp2_csp.png`. All network results are simulated on the 2004 Abilene topology and traffic.

## Experiment 1: BFS vs UCS vs A*

Abilene: all 132 ordered pairs. germany50: 100 random ordered pairs, 5 seeds for the sample (spread = std over seeds);
every timing is the mean of 5 repeats. Counters follow the README definitions.

| Graph | Algorithm | Expanded | Generated | Path cost (km) | Hops |
|---|---|---|---|---|---|
| Abilene | BFS | 7.0 | 8.9 | 2308.9 | 2.50 |
| Abilene | UCS | 7.0 | 9.1 | 2210.9 | 2.59 |
| Abilene | A* | 4.0 | 6.8 | 2210.9 | 2.59 |
| germany50 | BFS | 26.6 ± 1.1 | 33.6 ± 1.1 | 426.9 ± 14.6 | 4.09 ± 0.17 |
| germany50 | UCS | 26.6 ± 0.8 | 36.7 ± 1.1 | 378.9 ± 14.0 | 4.54 ± 0.14 |
| germany50 | A* | 7.6 ± 0.5 | 16.7 ± 0.6 | 378.9 ± 14.0 | 4.54 ± 0.14 |

- **A\* matches UCS in cost and never expands more nodes.** The path cost is identical in 100% of pairs on both graphs;
  A* expands fewer nodes in 86% (Abilene) and 97% (germany50) of pairs and the same number in the rest, never more. The
  mean expansion ratio A*/UCS is 0.62 on Abilene and 0.33 on germany50: the heuristic helps more on the larger graph.
- **BFS finds the fewest hops, not the shortest path.** Its hop count equals the unweighted shortest path in every pair
  (checked against networkx). Its path is longer than the optimum in 23% of Abilene pairs and 56% of germany50 pairs,
  by 17% and 20% when it is longer. It has fewer hops than the UCS path in 9% and 29% of pairs.
- **Wall time is not the story on these tiny graphs.** One search takes 4 to 40 microseconds. A* is slower than BFS and, on
  Abilene, than UCS in time, because computing the haversine heuristic costs more than it saves on a 12-node graph;
  on germany50 A* is faster than UCS (0.026 vs 0.040 ms). The node counts are the reliable comparison.

## Experiment 2: BT vs BT+FC vs BT+FC+AC-3

Instances: K in {10, 20, 40} x W in {4, 8} x 10 test hours (504 + 17 i), top-K pairs by NN-forecast demand (seed 0),
3 routes per request within 5,000 km. Every method gets the identical instance, `precheck` is off, the cap is 1,000,000
assignments. A true-demand run and a small-W run (W = 1, 2) are in `exp2_csp_summary.md`. Every solution passes
`is_valid`, and methods that finish always agree on solvable or not.

| K | W | solved / unsolvable / timeout | BT assignments | BT+FC assignments | BT+FC+AC-3 assignments |
|---|---|---|---|---|---|
| 10 | 4 | 1 / 9 / 0 | 22,818 ± 31,046 | 7,703 ± 8,717 | 7,703 ± 8,717 |
| 10 | 8 | 10 / 0 / 0 | 10 | 10 | 10 |
| 20 | 4 | 0 / 10 / 0 | 47,026 ± 74,813 | 13,109 ± 18,210 | 13,109 ± 18,210 |
| 20 | 8 | 10 / 0 / 0 | 20 | 20 | 20 |
| 40 | 4 | 0 / 10 / 0 | 47,026 ± 74,813 | 12,972 ± 18,010 | 12,972 ± 18,010 |
| 40 | 8 | 0 / 0 / 10 | ≥1,000,000 | ≥1,000,000 | ≥1,000,000 |

- **Forward checking helps exactly where there is something to prune.** On the instances where the search has to
  backtrack (W = 4), FC needs 3.0x (K = 10) to 3.6x (K = 20) fewer assignments than plain BT, and the like-for-like median is 1.5x to 2.5x
  over both demand sources. When W = 8 and the instance is easy, all three methods make exactly K assignments
  (no backtracking) and are indistinguishable.
- **AC-3 prunes nothing when W >= 2.** In all 40 instances at W = 4 and 8 it removes 0.0% of the values: with at least two
  wavelengths, every value of a request has a supporting value of its neighbour on another wavelength. It only acts at
  W = 1, where it removes 100% of the values and proves the instance unsolvable with 0 assignments (BT needs 4 and FC 2).
  So BT+FC+AC-3 behaves as BT+FC at the plan's settings (the plan anticipated "may prune little").
- **K = 20 and K = 40 at W = 4 look the same because the contradiction is among the top requests.** Variables are ordered by
  predicted demand, so the search fails inside the first few variables and never reaches the low-priority ones. The
  planner then blocks about 12 of 20 (K = 20) and 32 of 40 (K = 40) requests (feedback loop, last column of the full table).
- **K = 40, W = 8 hits the cap in all 10 hours for all three methods.** The instance is unsolvable in these hours (ten
  or more requests are forced through one link, with only 8 wavelengths), but the search cannot see this and runs to the
  cap. The `precheck` option proves it instantly with 0 assignments. FC is about 5x faster than BT per
  assignment at the cap (about 2.1 s vs 11.7 s per run) because it avoids re-checking every earlier neighbour.
- **W = 4 is nearly always infeasible with the full route set in this setting:** only 1 of 10 hours is solvable at K = 10 and none at K = 20 or 40 (3 of 10 with true demand at K = 10). This matters for Experiment 3: its settings need to
  allow the planner to serve a useful share (K = 10 or W = 8).

## Limits of these results

- 10 hours is a small sample; the standard deviations are large because instance difficulty varies a lot between hours.
- Timeout rows are lower bounds, not means; plain BT's counts at K = 40, W = 8 are only the cap.
- Search times are microseconds and CPU-dependent. Counters are the primary measure.
