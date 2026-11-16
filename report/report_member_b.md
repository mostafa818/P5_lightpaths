# Report, Member B sections (draft for Member A's review)

Sections for Member B's code: design choices for search and CSP, search, CSP, and Experiments 1 and 2. Budget in the plan:
design choices 1.5 pages (shared), search 1.5, CSP 2, experiments 2 (shared). All network results are **simulated**.

## 1. Search

**Graph.** Adjacency dict `node -> [(neighbour, length_km)]`. A link id is the sorted pair of its two node names, so A-B and B-A
are the same cable; this matters later for wavelength conflicts. Abilene has 12 nodes and 15 links; germany50 (Experiment 1
only) has 50 nodes and 88 links. Link cost is the haversine distance between the endpoints.

**Algorithms.**
- *BFS:* FIFO queue, visited set, parent pointers. Finds the fewest hops, not the shortest distance; the reported cost is the sum
  of the link lengths of the path found.
- *UCS:* priority queue ordered by path cost g; stops when the goal is **popped**, which is what guarantees optimality for
  non-negative costs; a stale entry of a closed node is discarded.
- *A\*:* same function as UCS with f = g + h, where h is the haversine distance to the destination. Link costs equal straight-line
  distances, so h never overestimates and is consistent: A\* is optimal and never expands more nodes than UCS. UCS is A\* with h = 0.

**Counters.** *Generated* = every node pushed onto the frontier (the source counts; a cheaper re-push counts again).
*Expanded* = every node popped and processed (the goal counts, searches stop at pop). All three algorithms count identically.

**Routes per request (Yen's algorithm).** Route 1 is the shortest path. Each next route is found by taking each earlier route,
removing the edges already used from each spur node (and the earlier nodes of the path), searching from the spur node to the
destination, joining the root path and keeping the cheapest unused candidate. Routes longer than the reach limit (5,000 km,
inclusive) are dropped. Simple paths are loop-free by construction. `src == dst` raises an error; fewer than k paths return fewer
routes; an unreachable node returns an empty list.

**Tests.** A hand-made 5-node graph with a known shortest path; UCS and A\* costs equal to `networkx.shortest_path_length` on all
132 Abilene pairs; Yen's routes compared with `networkx.shortest_simple_paths`; the germany50 loader checked against its `dist`
values.

**Demo trace.** `python run_demo.py --trace` prints expansion order for one pair. For CHINng to LOSAng both find the same
3,922 km path, UCS expanding 12 nodes and A\* 7; for the default pair KSCYng to LOSAng it is 12 against 4.

## 2. CSP

**Model.** Variables are the requests, ordered statically by predicted demand (highest first). A value is a pair (route, wavelength)
with wavelength 0 to W-1; value order is shortest route first, then lowest wavelength. Two assignments clash if their routes share
a link and use the same wavelength. Which route pairs share a link is precomputed once (bitmasks of link ids), because the check
runs millions of times; two requests are neighbours only if some pair of their routes shares a link.

**Algorithms (switchable).**
- *Plain backtracking (BT):* assign in order, check consistency against assigned requests only, count an assignment for each value
  that passes, and count a backtrack when a variable has no value left.
- *Forward checking (FC):* after assigning, delete from each unassigned neighbour every value that clashes; an emptied domain causes
  an immediate undo. Deleted values are restored from a trail on backtrack.
- *AC-3:* arc consistency run once before the search (BT+FC+AC-3 means AC-3 as preprocessing and FC inside the search): remove every value
  with no compatible value in a neighbour, re-queue affected arcs, stop at an empty queue; an empty domain proves no solution.
- *Safety cap:* `max_assignments` = 1,000,000; reaching it returns status `timeout` with the counters.
- *Precheck (optional):* reports at once an instance with more requests forced through one link than there are wavelengths. It never
  changes the answer, only the speed; the integration uses it, Experiment 2 does not.

**Validator and messages.** `is_valid` checks every pair of assigned requests for a clash and the reach limit; every test and
experiment uses it. Impossible cases return `None` with `No solution: wavelengths exhausted on link X-Y` or
`No solution: no route within reach for request R`.

**Tests.** Hand-made instances with known answers; many small random instances on which BT, BT+FC and BT+FC+AC-3 must agree on
solvable or not and every solution must pass `is_valid`; requests that share a link get different wavelengths while link-disjoint
requests may share one; edge cases K = 1, W = 1 with disjoint requests, reach exactly equal to a route length, tied demands; the two
impossible cases above.

## 3. Experiment 1: BFS vs UCS vs A\*

Abilene: all 132 ordered pairs. germany50: 100 random ordered pairs x 5 seeds (spread = std over seeds); timings are means of 5 repeats.

| Graph | Algorithm | Expanded | Generated | Path cost (km) | Hops |
|---|---|---|---|---|---|
| Abilene | BFS | 7.0 | 8.9 | 2308.9 | 2.50 |
| Abilene | UCS | 7.0 | 9.1 | 2210.9 | 2.59 |
| Abilene | A\* | 4.0 | 6.8 | 2210.9 | 2.59 |
| germany50 | BFS | 26.6 ± 1.1 | 33.6 ± 1.1 | 426.9 ± 14.6 | 4.09 ± 0.17 |
| germany50 | UCS | 26.6 ± 0.8 | 36.7 ± 1.1 | 378.9 ± 14.0 | 4.54 ± 0.14 |
| germany50 | A\* | 7.6 ± 0.5 | 16.7 ± 0.6 | 378.9 ± 14.0 | 4.54 ± 0.14 |

- A\* has the same cost as UCS in every pair and never expands more nodes: fewer in 86% (Abilene) and 97% (germany50) of pairs; mean
  expansion ratio A\*/UCS 0.62 and 0.33, so the heuristic helps more on the larger graph.
- BFS finds the fewest hops but a longer path in 23% (Abilene) and 56% (germany50) of pairs, by 17% and 20% when longer.
- Wall time is not the story on graphs this small (4 to 40 microseconds per search); the haversine heuristic costs more than it
  saves on 12 nodes, and A\* is faster than UCS only on germany50 (0.026 vs 0.040 ms). Node counts are the reliable comparison.

Figure: `outputs/experiments/exp1_search.png`. Raw rows: `exp1_search_raw.csv`.

## 4. Experiment 2: BT vs BT+FC vs BT+FC+AC-3

K in {10, 20, 40} x W in {4, 8} x 10 test hours (504 + 17 i); top-K pairs by NN forecast (seed 0), 3 routes per request within 5,000 km.
All methods get the identical instance, precheck off, cap 1,000,000 assignments. Spread = std over the 10 hours, not over seeds.

| K | W | solved / unsolvable / timeout | BT assignments | BT+FC assignments | BT+FC+AC-3 assignments |
|---|---|---|---|---|---|
| 10 | 4 | 1 / 9 / 0 | 22,818 ± 31,046 | 7,703 ± 8,717 | 7,703 ± 8,717 |
| 10 | 8 | 10 / 0 / 0 | 10 | 10 | 10 |
| 20 | 4 | 0 / 10 / 0 | 47,026 ± 74,813 | 13,109 ± 18,210 | 13,109 ± 18,210 |
| 20 | 8 | 10 / 0 / 0 | 20 | 20 | 20 |
| 40 | 4 | 0 / 10 / 0 | 47,026 ± 74,813 | 12,972 ± 18,010 | 12,972 ± 18,010 |
| 40 | 8 | 0 / 0 / 10 | ≥1,000,000 | ≥1,000,000 | ≥1,000,000 |

- **Forward checking helps where there is something to prune:** at W = 4, 3.0x (K = 10) to 3.6x (K = 20) fewer assignments than BT. On
  easy instances (W = 8) all methods make exactly K assignments and are indistinguishable.
- **AC-3 prunes nothing at W >= 2** (0.0% of values in all 40 instances at W = 4 and 8); it only acts at W = 1, where it removes 100%
  of the values and proves infeasibility with 0 assignments (BT needs 4, FC 2). BT+FC+AC-3 therefore behaves as BT+FC at these settings.
- **K = 20 and K = 40 at W = 4 look alike** because the contradiction is among the top requests, so the search fails inside the first
  few variables; the planner then blocks about 12 of 20 and 32 of 40 requests.
- **K = 40, W = 8 reaches the cap in all 10 hours** for all three methods: the instance is unsolvable (ten or more requests forced
  through one link, 8 wavelengths) but the search cannot see this; precheck proves it with 0 assignments. FC is about 5.5x faster
  than BT per run at the cap (2.1 s vs 11.7 s in one run, 1.8 s vs 10.6 s in an independent rerun; absolute times are machine-dependent).
- **W = 4 is nearly always infeasible** with the full route set here: 1 of 10 hours solvable at K = 10 and none at K = 20 or 40. This is why
  Experiment 3 reports W = 4 (hard) and W = 8 (easy).

Figure: `outputs/experiments/exp2_csp.png`; CSP domain sizes: `outputs/csp_domains.png`. Raw rows: `exp2_csp_raw.csv`.

## 5. Limits that concern Member B's part

- 10 hours and one NN seed in Experiment 2; large standard deviations because instance difficulty varies by hour.
- Timeout rows are lower bounds, not means.
- Search times are microseconds and CPU-dependent; counters are the primary measure.
- Abilene has 15 links, so extra routes rarely help and AC-3 pruned nothing; results may differ on larger or denser networks. germany50
  was used for Experiment 1 only.
- Static variable order (highest demand first) follows the brief; dynamic orders such as minimum remaining values were not evaluated.
