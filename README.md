# P5 Lightpaths: Route and Wavelength Assignment (Search + CSP + Neural Network)

A small planning pipeline for an optical backbone, run on the 12-node Abilene network:

1. **Forecast.** A neural network predicts the next hour's traffic demand for all 132 ordered router pairs.
2. **Requests.** The K pairs with the highest predicted demand become lightpath requests (predicted demand = priority).
3. **Search.** For each request up to 3 loop-free routes are found with Yen's algorithm on top of UCS or A\*.
4. **CSP.** Every request gets one route and one wavelength so that no two requests share a link on the same wavelength
   (backtracking, forward checking, AC-3).
5. **Feedback loop.** If no assignment exists, retry with more routes, then drop the lowest-priority request and solve again.

**All network results are simulated** on 2004 measurements; nothing was run on a real optical network (see
[Limitations](#limitations)). Planning the same instance with the NN forecast and with a naive last-hour forecast shows
whether the better forecast helps the planner (Experiment 3). Short answer: it does not (see [Results](#results)).

## Contents
[Quick start](#quick-start) | [Configuration](#configuration) | [Data](#data) | [How it works](#how-it-works) |
[Counter definitions](#counter-definitions) | [Demo](#demo) | [Experiments and results](#experiments-and-results) |
[Tests](#tests) | [Reproducibility](#reproducibility) | [Limitations](#limitations) | [Repository layout](#repository-layout) |
[Team and workflow](#team-and-workflow)

## Quick start
Python 3.12. The NN library is scikit-learn (`MLPRegressor`).

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # pinned versions
pip install -e .                                       # puts src/ on the import path, no PYTHONPATH needed
# put the raw data in data/raw/ (see "Getting the raw data" below)
python make_data.py                                    # raw files -> data/processed/hourly.npz + meta.json (seconds)
python run_demo.py                                     # one planning run: input, routes, assignment, blocked list
pytest                                                 # all tests (about 15 s)
python -m eval.run_experiments                         # Experiments 1, 2, 3 (about 20 minutes in total)
```

What is committed, so a fresh clone is usable even before the raw data is copied in: the processed dataset
(`data/processed/`), the trained NN models (`outputs/nn/`), all experiment results and figures (`outputs/`), and cached
demo output (`outputs/demo_cache/`). What is **not** committed (git-ignored): the raw traffic and topology files in
`data/raw/`. Without them the tests that need raw data skip (each says so), `make_data.py` stops with a clear message,
and `run_demo.py` needs only the small topology file `data/raw/topology/abilene_sndlib_native.txt`.

Optional tool, not in `requirements.txt`: `pip install coverage` and
`coverage run --include="*/src/*,*/run_demo.py,*/make_data.py" -m pytest`.

## Configuration
Everything that is generated or chosen lives in `config.yaml`; [docs/data_dictionary.md](docs/data_dictionary.md)
states how each value was set. Experiment variables (K, W) are also set inside the experiment scripts, as noted.

| Key | Value | Meaning | Read by |
|---|---|---|---|
| `seed` | 42 | global seed (data pipeline, bootstrap, experiment sampling) | `make_data.py`, `eval/*` |
| `rwa.num_wavelengths` | 8 | W, default of `run_demo.py`; the experiments use W = 4 and W = 8 | `run_demo.py` |
| `rwa.top_k_requests` | 20 | K, default of `run_demo.py`; Experiment 2 uses K = 10, 20, 40 | `run_demo.py` |
| `rwa.reach_km` | 5000 | maximum length of a lightpath route; the longest shortest path is 4,706 km, so every pair has a route | `integration`, `search`, experiments |
| `rwa.routes_per_request` / `max_routes` | 3 / 5 | routes per request, and the larger number used on retry | `integration` |
| `rwa.lightpath_capacity_mbps` | 9920 | OC-192 rate. Assumption check only: one lightpath carries a whole request; a test asserts every hourly demand fits | `tests` |
| `csp.max_assignments` | 1,000,000 | safety cap per solve; a solve that reaches it returns status `timeout` | `csp`, `integration`, experiments |
| `nn.*` | seeds 0-4, hidden [64, 64], lr 1e-3, batch 256, 100 epochs, patience 10 | MLP training; early stopping on the validation MSE (log scale) | `nn/train.py` |
| `split.*` | train hours 0-503, test 504-671, validation = last 3 days of train | chronological split of the 672 hours | `make_data.py`, `nn` |
| `node_order` | 12 router names, alphabetical | row/column order of the traffic matrices (verified, see Data) | `make_data.py`, `integration` |
| `paths.*` | raw, processed, outputs, topology files | file locations | all |

## Data
| Item | Source | Notes |
|---|---|---|
| Traffic (used) | Yin Zhang's Abilene traffic matrices: weekly files `X06` to `X09` (2004-05-01 to 2004-05-28) and the routing matrix `A` | observational; 5-minute matrices, unit 100 bytes / 5 minutes |
| Abilene topology | `abilene_sndlib_native.txt`, SNDlib native format | 12 nodes, 15 links, longitude/latitude per node; a copy of the SNDlib file from a public GitHub mirror |
| germany50 topology | `germany50.json`, a TopoHub copy of the SNDlib network | 50 nodes, 88 links; used only in Experiment 1 |
| Mapping cross-check | `demandMatrix-abilene-zhang-5min-20040301-0145.txt`, SNDlib sample | same Zhang data under SNDlib node names |

Citations for the report: SNDlib (S. Orlowski, R. Wessaly, M. Pioro, A. Tomaszewski, "SNDlib 1.0 - Survivable Network
Design Library", Networks 55(3), 2010) and Yin Zhang's Abilene traffic-matrix dataset. The SNDlib download links
returned 404 errors when the project started, so the topology files are copies; this is stated in the report.

**Getting the raw data.** `data/raw/` is git-ignored. Download Yin Zhang's Abilene files (`A`, `X06.gz` to `X09.gz`,
`X01.gz`, `readme.txt`; the dataset page is believed to be cs.utexas.edu/~yzhang/research/AbileneTM/, not re-verified
from the build machine) into `data/raw/`, and the topology files into `data/raw/topology/`. The team's copy is on the
branch `raw-data` (never merge it into `main`): `git checkout origin/raw-data -- raw` and move the files. `make_data.py`
stores a SHA-256 of every raw file in `data/processed/meta.json`, so a rerun can be checked against the original data.

| Raw file | Size (bytes) | Content |
|---|---|---|
| `X06.gz` / `X07.gz` / `X08.gz` / `X09.gz` | 7,496,632 / 7,493,819 / 7,478,898 / 7,331,758 | weeks starting 2004-05-01 / 05-08 / 05-15 / 05-22 (used) |
| `X01.gz` (week 2004-03-01), `X10.gz` | 7,320,862 / 7,333,278 | cross-check and spare week |
| `A.txt` | 22,033 | routing matrix (link, demand, fraction); 144 demands in alphabetical sender-major order |
| `readme.txt`, `links`, `demands`, `topo-2003-04-10.txt` | 2,671 / 1,038 / 2,672 / 1,587 | Zhang's documentation and topology |

**What the pipeline does** (`make_data.py`, `src/data/pipeline.py`):
- Each `X*.gz` has 2,016 lines (12 x 24 x 7 five-minute matrices) of 720 values = 144 pairs x 5 blocks. Block 0 is the
  real measurement; the other four are gravity-model estimates and are not used.
- Mbps = value x 100 x 8 / (300 x 1e6). Each block of 12 five-minute values becomes one hourly mean; the 12 self-pairs are
  dropped, leaving 132 ordered pairs. Four consecutive weeks give an array of shape (672, 132).
- Chronological split: hours 0-503 train (the last 3 days are validation), hours 504-671 test. Hour 0 is 2004-05-01
  00:00, a Saturday. Lag features need 168 hours of history, so the first usable target hour is 168.
- **Node mapping verified** (the Week-1 risk): `A.txt` lists the demands in alphabetical sender-major order; its 15 links
  equal the SNDlib links; slot 21 (01:45) of `X01` against the SNDlib sample matrix gives Spearman 1.000 (0.80 if
  transposed), so rows are senders; the median ratio 2.67e-6 confirms the unit formula; paths implied by `A.txt` match
  distance-shortest paths for 114 of 132 pairs (the routers use link weights, not distances).
- Sanity checks: no NaN, no negatives, no all-zero hours or pairs, weekly boundaries continuous (ratio 0.95 to 0.97).
  Traffic is bursty: single pairs reach 5 Gbps for an hour against a per-pair median of 11 Mbps.

## How it works
`src/common.py` holds the interface contract shared by the two halves (`Request`, `Route`, `k_routes`, `solve`,
`forecast`). Each function there delegates to the module that implements it.

**Neural network** (`src/nn/`). 29 inputs per (target hour t, pair): demand at t-24 to t-1, demand at t-168, and sine and
cosine of hour of day and weekday. `log1p`, then standardised with statistics from the training hours only. MLP with two
hidden layers of 64 ReLU units, Adam, batch 256, MSE on the log scale, early stopping on the chronological validation
days, 5 seeds. `forecast(t, seed=0)` uses only data up to t-1 and returns 132 non-negative values. Baselines: last hour
(t-1) and same hour last week (t-168). Train and evaluate: `python -m nn.train`.

**Search** (`src/search/`). `topology.py` loads Abilene and germany50 and computes haversine link lengths; `graph.py` is an
adjacency-dict graph with undirected link ids (A-B and B-A are the same cable); `algorithms.py` has BFS, UCS and A\*
(UCS is A\* with h = 0; h is the haversine distance to the destination, which is consistent, so A\* is optimal);
`routes.py` is Yen's algorithm (`k_routes`, reach limit inclusive); `trace.py` prints expansion orders.

**CSP** (`src/csp/`). Requests are variables, ordered by predicted demand (highest first); a value is (route, wavelength),
shortest route first, then lowest wavelength. `model.py` precomputes which routes share a link (bitmasks) and has the
validator `is_valid`; `solver.py` is backtracking with optional forward checking (with a trail) and AC-3 as preprocessing,
a safety cap, and an optional `precheck` that reports at once an instance with more requests forced through one link
than there are wavelengths (it never changes the answer, only the speed; the integration uses it, Experiment 2 does not).
Messages: `No solution: wavelengths exhausted on link X-Y`, `No solution: no route within reach for request R`.
`analysis.py` computes domain sizes for the demo figure.

**Integration** (`src/integration/pipeline.py`). `plan(planner, t, predicted, K, W, method, strict)`: top-K requests,
`k_routes` with k = 3, `solve`. On failure: retry with k = 5 (skipped if the route lists are unchanged), then drop the lowest
predicted-demand request ("wavelengths exhausted", or "solver timeout" if the solve hit the cap) and solve again, until a
solution exists. Requests with no route within reach are blocked up front. **Strict mode** never drops anything and
returns the CSP's message. Every solve is logged, and every returned assignment is re-checked with `is_valid`.
`evaluate(result, true_demand)` gives **served share** (true demand of the true top-K pairs that were selected and
assigned, divided by the true demand of all true top-K pairs), **blocking rate** (blocked / K) and **top-K hit rate**.

## Counter definitions
Both halves count in exactly the same way, so Experiments 1 and 2 are comparable.

**Search**
- *nodes generated*: every node pushed onto the frontier.
- *nodes expanded*: every node popped from the frontier and processed.
- Details (BFS, UCS and A\* all count the same way): the source counts as generated; the goal counts as expanded when it is
  popped (searches stop at pop, not at push); in UCS and A\* a node is pushed again when a cheaper way to it is found, and
  each such push counts as generated; a stale queue entry for a closed node is discarded and not counted as expanded.

**CSP**
- *assignments*: every tentative variable = value attempt.
- *backtracks*: every time a variable has no value left and the search returns.
- Details (BT, BT+FC and BT+FC+AC-3 all count the same way): an assignment is counted each time a value is actually assigned,
  that is, after it passed the consistency check (plain BT) or was still in the domain (forward checking); values rejected by
  the check are not counted. A value undone because forward checking emptied a neighbour's domain is not a backtrack; a
  backtrack is counted only when a variable runs out of values. AC-3 preprocessing counts neither. `max_assignments` stops the
  search before assignment number `max_assignments + 1` and returns status `timeout`.

## Demo
`run_demo.py` prints the predicted top-K list (with the actual demand of that hour, for comparison only), the feedback-loop
log, the routes, the assignment table, the blocked requests with reasons and a summary line.

```
python run_demo.py                                    # hour 540, K=20, W=8 (config defaults), FC+AC-3, NN forecast: all 20 served
python run_demo.py --W 4 --plots                      # feedback loop: 8 of 20 served; four figures
python run_demo.py --nn off --W 4                     # last-hour forecast instead of the NN
python run_demo.py --W 4 --strict                     # no dropping: prints the CSP no-solution message
python run_demo.py --trace [--src X --dst Y]          # UCS vs A* expansion trace, default KSCYng -> LOSAng (12 vs 4 expansions)
python run_demo.py --method bt|fc|fc+ac3 --routes all|chosen|none --hour H --K K --W W --seed S --outdir DIR
```
Figures (`--plots`, written to `outputs/` or `--outdir`): (a) the topology at real coordinates with every served route
coloured by wavelength; (b) a link x wavelength heatmap (cell = priority rank of the request; a clash fails an
assertion); (c) CSP domain sizes with and without forward checking and AC-3; (d) NN vs last-hour in the planner.
Cached output of the live-demo runs is in `outputs/demo_cache/` (hour 540, K=20, W=8 and W=4, NN off, and a second trace).

## Experiments and results
`python -m eval.run_experiments` regenerates everything (raw CSVs, summary tables, plots in `outputs/experiments/`);
name experiments to run a subset: `python -m eval.run_experiments exp1 exp3`. Experiment 1 takes seconds, Experiment 2 about
14 minutes (plain backtracking runs to the cap on the hardest instances), Experiment 3 about 6 minutes (resumable:
finished chunks are kept; `python -m eval.exp3_nn --reuse` rebuilds its summary from the CSV). Full discussion of
Experiments 1 and 2: [report/experiments_search_csp.md](report/experiments_search_csp.md).

### Results
**NN forecast (Stage 4), test week, 5 seeds**

| Method | MAE (Mbps) | RMSE (Mbps) | top-10 hit | top-20 hit | top-40 hit |
|---|---|---|---|---|---|
| last hour (t-1) | 4.418 | 54.28 | 0.899 | 0.898 | 0.925 |
| same hour last week (t-168) | 11.118 | 118.27 | 0.823 | 0.816 | 0.856 |
| NN (MLP 2 x 64) | 4.630 ± 0.220 | 62.65 ± 2.01 | 0.899 ± 0.002 | 0.901 ± 0.001 | 0.927 ± 0.001 |

The NN does **not** beat the last-hour baseline on MAE or RMSE (traffic is bursty and the loss is on the log scale). Its
top-K hit rate differs from last-hour by no more than about 1.5 standard errors (paired per-hour differences, `outputs/nn_eval.json`).
No tuning on the test set was done.

**Experiment 1: BFS vs UCS vs A\*** (Abilene: all 132 pairs; germany50: 100 random pairs x 5 seeds)

| Graph | Algorithm | Expanded | Generated | Path cost (km) | Hops |
|---|---|---|---|---|---|
| Abilene | BFS | 7.0 | 8.9 | 2308.9 | 2.50 |
| Abilene | UCS | 7.0 | 9.1 | 2210.9 | 2.59 |
| Abilene | A\* | 4.0 | 6.8 | 2210.9 | 2.59 |
| germany50 | BFS | 26.6 ± 1.1 | 33.6 ± 1.1 | 426.9 ± 14.6 | 4.09 ± 0.17 |
| germany50 | UCS | 26.6 ± 0.8 | 36.7 ± 1.1 | 378.9 ± 14.0 | 4.54 ± 0.14 |
| germany50 | A\* | 7.6 ± 0.5 | 16.7 ± 0.6 | 378.9 ± 14.0 | 4.54 ± 0.14 |

A\* has the same cost as UCS in every pair and never expands more nodes: it expands fewer in 86% of the Abilene pairs and
97% of the germany50 pairs (mean expansion ratio A\*/UCS 0.62 and 0.33). BFS finds the fewest hops but a longer path in 23% (Abilene) and 56% (germany50) of the pairs.

**Experiment 2: BT vs BT+FC vs BT+FC+AC-3** (10 test hours, same instances for all methods, NN-forecast instances, cap
1,000,000; mean assignments; full table in `outputs/experiments/exp2_csp_summary.md`)

| K | W | solved / unsolvable / timeout | BT | BT+FC | BT+FC+AC-3 |
|---|---|---|---|---|---|
| 10 | 4 | 1 / 9 / 0 | 22,818 | 7,703 | 7,703 |
| 20 | 4 | 0 / 10 / 0 | 47,026 | 13,109 | 13,109 |
| 40 | 4 | 0 / 10 / 0 | 47,026 | 12,972 | 12,972 |
| 10 | 8 | 10 / 0 / 0 | 10 | 10 | 10 |
| 20 | 8 | 10 / 0 / 0 | 20 | 20 | 20 |
| 40 | 8 | 0 / 0 / 10 | ≥ 1,000,000 | ≥ 1,000,000 | ≥ 1,000,000 |

Forward checking needs about 3.0x (K=10) to 3.6x (K=20) fewer assignments than plain backtracking at W=4, and is about
5x faster per assignment. AC-3 prunes nothing at W >= 2 on these instances (0.0% of values); it only matters at W = 1.
At K=40, W=8 every method hits the cap in all 10 hours (more than 8 requests are forced through one link).

**Experiment 3: NN vs last-hour forecast inside the planner** (168 test hours x 5 NN seeds, K=20)

| | metric | last-hour | NN (mean ± std over 5 seeds) | NN minus last-hour, paired | 95% block-bootstrap CI |
|---|---|---|---|---|---|
| W=4 | served share | 0.598 | 0.597 ± 0.002 | -0.0006 | [-0.0075, +0.0108] |
| W=4 | blocking rate | 0.581 | 0.587 ± 0.002 | **+0.0059** | **[+0.0013, +0.0093]** |
| W=4 | top-K hit rate | 0.898 | 0.901 ± 0.001 | +0.0029 | [-0.0026, +0.0085] |
| W=8 | served share | 0.943 | 0.944 ± 0.002 | +0.0011 | [-0.0023, +0.0044] |
| W=8 | blocking rate | 0.000 | 0.000 | 0 | 0 |

The NN gives **no practical gain** over the last-hour forecast. The one interval that excludes zero goes against the NN: at
W=4 it blocks about 0.6 percentage points more requests. The paired standard error treats the 168 hours as independent;
hourly traffic is autocorrelated, so the 24-hour block bootstrap (only 7 blocks, 2,000 resamples) is more reliable but still
rough. The top-K hit rate does not depend on W, so its two rows are one comparison; with the zero blocking at W=8 that leaves
four distinct comparisons, and a borderline interval should be read with caution. Likely reasons, not tested: a 4-week series
is short, last hour is a strong predictor of a slowly varying aggregate, and the NN is trained on the log scale while the
planner cares about the largest pairs.

**Integration findings**
- On Abilene with a 5,000 km reach, retrying with 5 routes never fixed a failed 3-route solve (0 of 2,688 real instances
  scanned: K from 5 to 40, W from 2 to 8, 56 test hours, strict mode); the topology has only 15 links. The test showing
  "more routes fix a failure" therefore uses a toy network with five link-disjoint routes. The case where a request must be
  dropped is real (NN forecast, hour 540, K=20, W=4).
- The brief's drop rule (always drop the globally lowest-priority request) is conservative: with the static
  highest-demand-first order the unsolvable core is often among the top requests, so dropping low-priority ones repeats an
  identical failing search until the prefix that fits is left (hour 540, K=20, W=4: 8 of 20 served after 13 failing solves
  with identical counts). It is implemented as the brief describes; dropping the lowest-priority request involved in the
  conflict is a possible variant and was not run.

## Tests
`pytest` runs everything (238 tests with the raw data present, about 15 s). In a fresh clone **without** the raw data:
178 passed, 60 skipped, 0 failed; the skipped tests need the git-ignored raw files and say so. The plan's minimum
(5 normal, 3 edge, 2 impossible) is exceeded and mapped test by test in [docs/test_matrix.md](docs/test_matrix.md); that file
also lists the test for every plan step, and a meta-test (`test_matrix.py`) fails if a named test disappears. Every
requirement of the plan minimum in it (normal, edge, impossible) has a test that passes without the raw data. Coverage of the data, NN, integration and demo code is 99%
(only `__main__` guards are uncovered). Run with `pytest -rs` to see why a test skipped.

## Reproducibility
- Fixed seed 42 and pinned versions (`requirements.txt`, Python 3.12). Raw files are only read, never edited.
- From a fresh clone with the raw data, `make_data.py`, `python -m nn.train` and `python -m eval.exp3_nn --reuse`
  regenerate the committed files byte for byte (`git status` stays clean). The committed `outputs/nn/*.joblib` models load
  only with the pinned scikit-learn (1.9.1); if you change versions, regenerate them with `python -m nn.train` (about 10 s).
- Experiment counters are deterministic; timings vary with the machine, so the experiment tables report them as means over
  repeats and the write-up gives ratios, not absolute times, as the result.

## Limitations
**Required by the brief**
- *Observational 2004 data.* The traffic was measured on Abilene in May 2004 and is reused as if it were a recent network;
  traffic patterns and optical networks have changed a great deal since.
- *No traffic destined for the Abilene AS.* The data does not include traffic destined for the Abilene network itself, so the
  demands are an incomplete picture of the load.
- *Wavelength continuity.* A lightpath keeps one wavelength end to end (no wavelength converters), which is what makes this a
  CSP; networks with converters would be easier.
- *One lightpath per request.* A request is served by a single lightpath on a single route (no splitting over routes, no
  grooming). Every hourly demand fits in one lightpath (largest 5.2 Gbps against 9,920 Mbps), which a test asserts.
- *Hourly forecasts.* The horizon is one hour; the network is planned for the hourly mean, not for 5-minute peaks.
- *Everything simulated.* Routes, wavelengths and blocking are computed on a model of the network; no real equipment, no
  signal-quality or physical-layer model beyond a fixed reach limit.

**Found while building and testing this project**
- Only four weeks of data and a single test week: the Experiment 3 intervals rest on 168 correlated hours (7 daily blocks).
- The NN does not improve on a last-hour forecast; conclusions about "a better forecast" apply to this NN and this series.
- Experiment 2 uses 10 test hours and one NN seed, so its spread is across hours. At K=40, W=8 all methods reach the cap, so
  those rows are lower bounds.
- Abilene has 15 links, so extra routes rarely help; results (and the AC-3 finding) may differ on larger or denser networks.
  germany50 was used for Experiment 1 only; the full pipeline was run on Abilene only.
- The reach limit (5,000 km) and the retry route count (5) are chosen values, not measured physical limits.
- The brief's drop rule is conservative (see Integration findings); the alternative was not evaluated.
- The topology files are copies from mirrors of SNDlib, not downloads from SNDlib itself; the dataset URL is believed
  correct but was not re-verified from the build machine.

## Repository layout
```
config.yaml            all parameters (see Configuration)
make_data.py           raw files -> data/processed/
run_demo.py            one-command demo (see Demo)
requirements.txt       pinned versions;  pyproject.toml, pytest.ini: editable install and pytest paths
src/common.py          interface contract (Request, Route, k_routes, solve, forecast)
src/data/              hourly pipeline
src/nn/                features, model, evaluation, training, forecast(t)
src/search/            topology, graph, BFS/UCS/A*, Yen's routes, expansion trace
src/csp/               model, solver (BT, FC, AC-3, precheck), validator, domain-size analysis
src/integration/       planner with the feedback loop, strict mode, metrics
src/eval/              plots, experiments 1-3, run_experiments.py
src/tests/             all tests
data/raw/              raw files (git-ignored);  data/processed/: hourly.npz, meta.json, abilene_links.csv
outputs/               figures, trained models (nn/), experiment results (experiments/), demo cache (demo_cache/)
docs/                  data_dictionary.md, test_matrix.md
report/                experiment write-up (experiments_search_csp.md)
```

## Team and workflow
Two members: **A** (data, NN, integration, demo, Experiment 3) and **B** (search, CSP, Experiments 1 and 2). Each worked on
`member-a/...` or `member-b/...` branches and merged through pull requests reviewed by the other; the review comments and the
fixes they led to are in the pull-request history. The git log is a deliverable: `git log --oneline --graph`.
