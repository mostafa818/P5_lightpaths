# P5 Lightpaths: Route and Wavelength Assignment (Search + CSP + NN)

## Setup
Python 3.12. NN library: scikit-learn (MLPRegressor).
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .      # puts src/ on the import path (no PYTHONPATH needed)
# when pinning: pip freeze --exclude-editable > requirements.txt
```
Commands (filled in as stages are completed):
```
python make_data.py
python run_demo.py ...
python -m eval.run_experiments            # all experiments; or: python -m eval.run_experiments exp1 exp2 exp3
pytest
```

## Data sources (decided; SNDlib download links returned 404)
| Item | Source | Notes |
|---|---|---|
| Abilene topology | `abilene_sndlib_native.txt` (SNDlib native format, copy from a public GitHub repo) | 12 nodes, 15 links, lon/lat per node |
| germany50 topology | `germany50.json` (TopoHub copy of SNDlib) | 50 nodes, 88 links, `dist` in km; Experiment 1 only |
| Traffic (primary) | Zhang weekly files `X06`-`X09` + routing matrix `A` | needs node-mapping check (Step 1.9) |
| Mapping cross-check | `demandMatrix-abilene-zhang-5min-20040301-0145.txt` (SNDlib sample) | compare with slot 21 of `X01` |
| Fallback traffic | SNDlib Abilene dynamic matrices | only if Zhang files unobtainable; availability unverified |

Raw files go in `data/raw/` (traffic) and `data/raw/topology/` (topologies); both are git-ignored. Cite SNDlib and Yin Zhang's Abilene data in the report and state the topology files are copies (TopoHub, GitHub mirror).

## Getting the raw data
`data/raw/` is git-ignored. Download Yin Zhang's Abilene traffic-matrix files (`A`, `X06.gz`-`X09.gz`, `X01.gz`, `readme.txt`; dataset page believed to be cs.utexas.edu/~yzhang/research/AbileneTM/ - not re-verified from the build sandbox) into `data/raw/`, and the topology files into `data/raw/topology/`. `make_data.py` stops with a clear message if a file is missing and stores a SHA-256 of each raw file in `data/processed/meta.json`, so a rerun can be checked against the original data. Project team copy: branch `raw-data` (never merge into main).

## Raw files (Step 1.1; never edited, only read)
| File | Size (bytes) | Content |
|---|---|---|
| `X06.gz` / `X07.gz` / `X08.gz` / `X09.gz` | 7,496,632 / 7,493,819 / 7,478,898 / 7,331,758 | weeks starting 2004-05-01 / 05-08 / 05-15 / 05-22 (used) |
| `X01.gz` (week 2004-03-01), `X10.gz` | 7,320,862 / 7,333,278 | cross-check, spare week |
| `A.txt` | 22,033 | routing matrix (link, demand, fraction); 144 demands in src-major alphabetical order |
| `readme.txt`, `links`, `demands`, `topo-2003-04-10.txt` | 2,671 / 1,038 / 2,672 / 1,587 | Zhang's documentation and topology |

## Data pipeline results (Steps 1.2-1.10)
- Each X file: 2,016 lines x 720 values (144 OD pairs x 5 blocks); block 0 is the real measurement, the other four are gravity-model estimates and are NOT used.
- Mapping verified (Step 1.9): `A.txt` names the demands in alphabetical src-major order; the 15-link topology in `links` equals the SNDlib Abilene links; X01 slot 21 vs the SNDlib 2004-03-01 01:45 sample gives Spearman 1.000 (transposed: 0.80), so rows = senders. Median ratio 2.67e-6 confirms the unit formula Mbps = v x 100 x 8 / (300 x 1e6). Paths implied by `A.txt` match the distance-shortest paths for 114 of 132 pairs (OSPF uses weights, not distance).
- Dataset: `data/processed/hourly.npz` (672 x 132, Mbps hourly mean), `meta.json`. Train hours 0-503, validation 432-503 (last 3 days of train), test 504-671. Hour 0 = 2004-05-01 00:00 (Saturday); keep this in mind for weekday features.
- Traffic is bursty (single pairs reach 5 Gbps for an hour against a per-pair median of 11 Mbps); weekly boundaries are continuous (first/last-hour ratio 0.95-0.97).

## Neural network results (Stage 4, test week = hours 504-671, 5 seeds, mean +- std)
Train: `python -m nn.train` (writes `outputs/nn/`, `outputs/nn_eval.json`). 29 features, 34,848 train / 9,504 validation / 22,176 test rows.

| Method | MAE (Mbps) | RMSE (Mbps) | top-10 hit | top-20 hit | top-40 hit |
|---|---|---|---|---|---|
| last hour (t-1) | 4.418 | 54.28 | 0.899 | 0.898 | 0.925 |
| same hour last week (t-168) | 11.118 | 118.27 | 0.823 | 0.816 | 0.856 |
| NN (MLP 2x64) | 4.630 +- 0.220 | 62.65 +- 2.01 | 0.899 +- 0.002 | 0.901 +- 0.001 | 0.927 +- 0.001 |

Honest reading: the NN does **not** beat the last-hour baseline on MAE or RMSE (traffic is bursty and the NN is trained on the log scale). There is **no measurable difference in top-K hit rate**: the paired per-hour difference NN minus last-hour is -0.0002 +- 0.0036 (K=10), +0.0029 +- 0.0024 (K=20) and +0.0020 +- 0.0015 (K=40) (SE over the 168 test hours; see `paired_hit_diff_vs_last_hour` in `outputs/nn_eval.json`), i.e. within about 1.5 standard errors of zero (the SE treats the 168 test hours as independent; hourly traffic is autocorrelated, so the true uncertainty is probably somewhat larger). The worse RMSE/MAE is probably because the loss is on the log scale (it favours relative error), while the planner cares about the large pairs: the true top-20 pairs carry about 61% of total demand. Experiment 3 will show whether the planner notices. No tuning on the test set was done. `forecast(t, seed=0)` in `src/nn/forecast.py` returns 132 non-negative predictions using only data up to t-1.

Note: the committed `outputs/nn/*.joblib` models only load with the pinned scikit-learn version in `requirements.txt` (1.9.1); regenerate them with `python -m nn.train` (about 10 s, bit-for-bit reproducible) if you change versions. `common.forecast(t)` delegates to `nn.forecast.forecast(t, seed=0)`.

## Integration (Stage 5): `src/integration/pipeline.py`
`plan(planner, t, predicted, K, W, method="fc+ac3", strict=False)`: forecast -> top-K requests (predicted demand = priority, `Request.id` = pair index) -> `k_routes` (k=3, reach limit) -> `solve(..., precheck=True)`; on failure retry with k=5, then drop the lowest-priority request ("wavelengths exhausted", or "solver timeout" if the solve hit the cap) and solve again until a solution exists. Requests with no route within reach are blocked up front. **Strict mode** never drops or pre-blocks and returns the CSP's message (`status` is `no_solution`, or `timeout` if the cap was hit; `evaluate` scores both as an unsolved hour). The k=5 retry is skipped when the k=5 route lists equal the k=3 lists (it would only repeat the failure), and after the first failure the longer route lists stay in use for later solves. Every solve is logged in `PlanResult.iterations` (`PlanResult.log_text()`), every returned assignment is re-validated with `csp.model.is_valid`. `evaluate(result, true_demand)` gives served share, blocking rate and top-K hit rate (true demand of the same hour). Forecast sources: `nn_forecast_fn(seed)` and `last_hour_forecast_fn(H)`.

Findings to report honestly:
- On Abilene with reach 5,000 km, retrying with k=5 never fixed a failed k=3 solve (0 of 2,688 real instances scanned: K in {5..40}, W in {2..8}, 56 test hours, strict mode); the topology has only 15 links. The "more routes fix it" evidence case (`test_evidence_more_routes_fix_a_failure`) therefore uses a toy network with five link-disjoint routes. The "a request must be dropped" case is real (`test_evidence_request_must_be_dropped_on_abilene`: NN forecast, hour 540, K=20, W=4).
- The plan's drop rule (always drop the globally lowest-priority request) is conservative: with a static highest-demand-first order the unsolvable core is often among the top requests, so dropping low-priority ones repeats an identical failing search until the prefix that fits is left (hour 540, K=20, W=4: 8 of 20 served, 13 failing solves that all report the identical counts; Member B suggested dropping the lowest-priority request *involved in the conflict* instead as a separately reported variant). It is implemented as the brief describes.

## Demo (Stage 6, Member A part): `run_demo.py`
```
python run_demo.py                                    # hour 540, K=20, W=8, FC+AC-3, NN forecast: all 20 served
python run_demo.py --W 4 --plots                      # feedback loop: 8 of 20 served, figures in outputs/
python run_demo.py --nn off --W 4                     # last-hour forecast instead of the NN
python run_demo.py --W 4 --strict                     # no dropping: prints the CSP no-solution message
python run_demo.py --method bt|fc|fc+ac3 --routes all|chosen|none --hour H --K K --W W --seed S --outdir DIR
python run_demo.py --trace [--src X --dst Y]          # UCS vs A* expansion trace (needs Member B's search.trace)
python -m eval.nn_effect --K 20 --W 4 8 --seeds 0     # NN vs last-hour inside the planner (data for figure d), ~2 min
```
It prints: the predicted top-K list (with the actual demand of that hour for comparison only), the feedback-loop log, the routes, the assignment table, the blocked requests with reasons and a summary line. Figures (`src/eval/plots.py`): (a) topology at real coordinates with every served route coloured by wavelength (colour-blind-safe Okabe-Ito palette, one lateral lane per wavelength so shared links stay readable), (b) link x wavelength heatmap (cell = priority rank of the request using it; a clash would fail an assertion), (d) NN vs last-hour bar chart with a paired-difference row. Plot (c), CSP domain sizes, and the trace are Member B's part (issue #7). Cached outputs for the live demo (plan 11.2) are in `outputs/demo_cache/` (text and figures for hour 540, K=20, W=8 and W=4, and NN off).

NN on vs off inside the planner: the full result (5 NN seeds) is Experiment 3, below. `python -m eval.nn_effect` is the underlying harness (1 seed by default, ~2 min); `run_demo.py --plots` draws figure (d) from `outputs/nn_effect.json`, which Experiment 3 writes.

## Experiment 3 (Member A): NN vs last-hour forecast inside the planner
`python -m eval.run_experiments exp3` (about 6 minutes; resumable, finished chunks are kept in `outputs/experiments/.exp3_chunks/`; `python -m eval.exp3_nn --reuse` rebuilds the summary and figure from `outputs/experiments/exp3_nn_raw.csv` and leaves an existing `outputs/nn_effect.json` untouched). Setup: 168 test hours x 5 NN seeds (and the deterministic last-hour forecast), K=20, planner BT+FC+AC-3 with precheck, cap 1,000,000 assignments, 3 routes (5 on retry), reach 5,000 km. Files: `outputs/experiments/exp3_nn_raw.csv`, `exp3_nn_summary.md`, `exp3_nn.png`.

| | metric | last-hour | NN (mean +- std over 5 seeds) | NN minus last-hour, paired | 95% block-bootstrap CI |
|---|---|---|---|---|---|
| W=4 | served share | 0.598 | 0.597 +- 0.002 | -0.0006 | [-0.0075, +0.0108] |
| W=4 | blocking rate | 0.581 | 0.587 +- 0.002 | **+0.0059** | **[+0.0013, +0.0093]** |
| W=4 | top-K hit rate | 0.898 | 0.901 +- 0.001 | +0.0029 | [-0.0026, +0.0085] |
| W=8 | served share | 0.943 | 0.944 +- 0.002 | +0.0011 | [-0.0023, +0.0044] |
| W=8 | blocking rate | 0.000 | 0.000 | 0 | 0 |
| W=8 | top-K hit rate | 0.898 | 0.901 +- 0.001 | +0.0029 | [-0.0026, +0.0085] |

Honest reading: the NN gives **no practical gain** over the last-hour forecast. Served share and top-K hit rate differ from last-hour by less than the uncertainty. The one difference whose interval excludes zero goes against the NN: at W=4 it blocks about 0.6 percentage points more requests. At W=8 nothing is blocked either way. The same-hour-last-week baseline is much worse still (Stage 4). The paired standard error treats the 168 hours as independent; hourly traffic is autocorrelated, so the 24-hour moving-block bootstrap interval (2,000 resamples, fixed seed) is the more reliable one. With 168 hours the 24-hour blocks give only 7 blocks, so even that interval is rough. The top-K hit rate depends on the forecast only (not on W), so its two table rows are one comparison, and blocking at W=8 is zero in every run: that leaves four distinct comparisons (served share at W=4 and W=8, blocking at W=4, hit rate), so a borderline interval such as the W=4 blocking one should be read with caution (\"slightly worse blocking\" is claimed, not \"the NN hurts\"). In the raw CSV the `assignments` and `backtracks` columns are totals over all solves of the feedback loop for one hour, not one solve: the 1,000,000 cap applies per solve (example: hour 641, last-hour forecast, W=4 totals 2.36 million assignments over 11 solves, the largest being 263,400, none a timeout; the CSV keeps only the totals). Likely reasons, not tested here: a 4-week series is short, last hour is a strong predictor for a slowly varying aggregate, and the NN is trained on the log scale while the planner cares about the largest pairs.

## Layout
`config.yaml` (all parameters) | `src/{data,nn,search,csp,integration,eval}/` | `src/common.py` (interface contract) | `src/tests/` | `data/{raw,processed}/` | `outputs/` | `report/` | `docs/`

## Counter definitions (both members MUST count this way)
**Search**
- *nodes generated*: every node pushed onto the frontier.
- *nodes expanded*: every node popped from the frontier and processed.
- Details used by `src/search/algorithms.py` (BFS, UCS and A* all count the same way): the source counts as generated; the goal counts as expanded when it is popped (searches stop at pop, not at push); in UCS/A* a node is pushed again whenever a cheaper way to it is found, and each such push counts as generated; a stale queue entry for an already closed node is discarded without counting as expanded.

**Search module** (`src/search/`): `topology.py` (Abilene and germany50 loaders, haversine link lengths), `graph.py` (adjacency-dict `Graph`, undirected link ids), `algorithms.py` (`bfs`, `ucs`, `astar`, all returning `(nodes, length_km)` or `None`), `routes.py` (Yen's algorithm, `k_routes`, reach limit inclusive). `common.k_routes` delegates to it.

**CSP**
- *assignments*: every tentative variable = value attempt.
- *backtracks*: every time a variable has no value left and the search returns.
- Details used by `src/csp/solver.py` (BT, BT+FC and BT+FC+AC-3 all count the same way): an assignment is counted each time a value is actually assigned to a variable, that is, after it passed the consistency check (plain BT) or was still in the domain (forward checking); values rejected by the check are not counted. A value undone because forward checking emptied a neighbour's domain is not a backtrack (the variable still has values to try); a backtrack is counted only when a variable runs out of values. AC-3 preprocessing counts neither. `max_assignments` stops the search before assignment number `max_assignments + 1` and returns status `timeout`.

**CSP module** (`src/csp/`): `model.py` (variables sorted by predicted demand, highest first, ties keep input order; bitmask domains per wavelength; precomputed route-conflict masks; `validate` / `is_valid`), `solver.py` (`solve(requests, routes_by_request, W, use_fc, use_ac3, max_assignments)`: backtracking with shortest-route-then-lowest-wavelength value order, forward checking with a trail, AC-3 as preprocessing, plus an optional `precheck`, off by default, that reports an instance with more requests forced through one link than wavelengths immediately with 0 assignments; it never changes the answer, only the speed; the integration uses `precheck=True`). The assignment maps request id to `(index into that request's route list, wavelength)`. A request with no route gives `No solution: no route within reach for request R`; when more requests are forced through one link than there are wavelengths the message is `No solution: wavelengths exhausted on link X-Y`. `common.solve` delegates to it.

## Git workflow
Branches `member-a/...` and `member-b/...`; merge via pull request reviewed by the other member; small, frequent commits.

## Limitations
(to be completed in Stage 9)

## Search trace and CSP domain sizes (Stage 6, Member B)
- `src/search/trace.py`: `expansion_trace(graph, src, dst, algo)` ("ucs", "astar", "bfs") returns the expansion order, path, cost and the same counters as Experiment 1; `format_trace(ucs, astar)` prints the two traces side by side. Good demo pair: `KSCYng -> LOSAng` (UCS expands 12 nodes, A* 4, same 2,762 km path).
- `src/csp/analysis.py`: `domain_sizes(requests, routes_by_request, W, fc_steps=1)` gives per request (priority order) the domain size initially, after AC-3 and after forward checking following the first `fc_steps` assignments. `python -m eval.plot_csp` writes `outputs/csp_domains.png` for hour 540, K=20, W=4 (forward checking prunes, AC-3 prunes nothing) and K=10, W=1 (AC-3 empties a domain: unsolvable with 0 assignments).
