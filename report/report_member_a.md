# Report, Member A sections (draft for Member B's review)

Sections for Member A's code: problem and data, neural network, integration, Experiment 3, and the limits that concern them.
Budget in the plan: problem and data 1.5 pages, NN 2, integration 1, share of experiments and limits. All network results are
**simulated** on 2004 Abilene measurements.

## 1. Problem and data

**Problem.** An optical backbone carries traffic as lightpaths. Each lightpath needs one route through the network and one
wavelength, and the same wavelength cannot be used by two lightpaths on the same fibre link (wavelength continuity, no
converters). The project plans the next hour in four steps: a neural network forecasts the demand of every router pair, the K
pairs with the highest forecast become lightpath requests, search finds up to 3 loop-free routes per request, and a constraint
satisfaction problem (CSP) assigns one route and one wavelength to every request. If no assignment exists, the planner retries
with more routes and then drops the lowest-priority request.

**Network.** Abilene as published in SNDlib: 12 nodes, 15 links, longitude and latitude per node. Link length is the haversine
distance between its endpoints (132 to 2,193 km, 14,029 km in total; longest shortest path 4,706 km). The reach limit is 5,000 km,
so every pair has a route. The topology files are copies from public mirrors of SNDlib, because the SNDlib download links
returned 404 errors when the project started.

**Traffic.** Yin Zhang's Abilene traffic matrices, weeks X06 to X09 (2004-05-01 to 2004-05-28), plus the routing matrix A. Each
weekly file has 2,016 five-minute matrices of 720 values = 144 pairs x 5 blocks. Only block 0 is the measurement; the other four
are gravity-model estimates and are not used. The unit is 100 bytes per 5 minutes, converted to Mbps. Twelve five-minute values
are averaged into one hourly rate, the 12 self-pairs are dropped, and four weeks give an array of shape (672 hours, 132 pairs).

**Node mapping (the main data risk).** The order of routers in the traffic files had to be verified. `A.txt` lists the demands in
alphabetical sender-major order and its 15 links equal the SNDlib links. Comparing slot 21 of `X01` with the SNDlib sample matrix
gives a Spearman correlation of 1.000 (0.80 if transposed), so rows are senders and the median ratio confirms the unit formula.
Shortest paths implied by `A.txt` match distance-shortest paths for 114 of 132 pairs; the remaining pairs differ because the
routers use link weights, not distances.

**Checks and split.** No NaN, no negative values, no all-zero hour or pair, weekly boundaries continuous (ratio 0.95 to 0.97).
Traffic is bursty: single pairs reach 5 Gbps for an hour against a per-pair median of 11 Mbps (maximum hourly demand
5,195.7 Mbps). The split is chronological: hours 0-503 train (the last 3 days are validation), hours 504-671 test. Figure:
`outputs/total_traffic_hourly.png` (daily rhythm over the four weeks). The data dictionary is `docs/data_dictionary.md`; it
separates real values (traffic), derived values (hourly means, link lengths) and generated values (W, K, reach, capacity).

## 2. Neural network forecast

**Task.** Predict the demand of all 132 pairs for hour t using only data up to t-1.

**Features (29 per target hour and pair).** Demand at t-24 to t-1, demand at t-168 (same hour last week), and sine and cosine of
hour of day and of weekday. `log1p` is applied to demands and the result is standardised with statistics from the training rows
only, so no test information leaks into training.

**Model.** MLP with two hidden layers of 64 ReLU units (scikit-learn `MLPRegressor`), Adam, learning rate 1e-3, batch 256, MSE on
the log scale, up to 100 epochs with early stopping on the chronological validation days. Five seeds (0 to 4). Predictions are
inverted to Mbps and clipped at 0. Tests check output shape, non-negativity, reproducibility for a fixed seed, and that no
feature uses hour t or later. No tuning was done on the test week.

**Baselines.** Last hour (t-1) and same hour last week (t-168).

**Results on the test week (5 seeds, mean ± std).**

| Method | MAE (Mbps) | RMSE (Mbps) | top-10 hit | top-20 hit | top-40 hit |
|---|---|---|---|---|---|
| last hour (t-1) | 4.418 | 54.28 | 0.899 | 0.898 | 0.925 |
| same hour last week (t-168) | 11.118 | 118.27 | 0.823 | 0.816 | 0.856 |
| NN (MLP 2 x 64) | 4.630 ± 0.220 | 62.65 ± 2.01 | 0.899 ± 0.002 | 0.901 ± 0.001 | 0.927 ± 0.001 |

The top-K hit rate (overlap between the K pairs ranked highest by forecast and by true demand, divided by K) matters more than
MAE, because it decides which requests enter the planner.

**Reading.** The NN does **not** beat the last-hour baseline on MAE or RMSE. Its top-K hit rate is within about 1.5 standard
errors of last hour (paired per-hour differences, `outputs/nn_eval.json`). Both are far better than same-hour-last-week. Likely
reasons, not tested: the series is short (4 weeks), the last hour is a strong predictor of a slowly varying aggregate, and the
loss is on the log scale while the planner cares about the largest pairs.

## 3. Integration

`plan(planner, t, predicted, K, W, method, strict)` in `src/integration/pipeline.py` chains the parts:

1. Take the top K predicted pairs as requests; predicted demand is the priority.
2. Call `k_routes` with k = 3 and the reach limit. A request with no route within reach is blocked at once
   ("no route within reach").
3. Call `solve` (BT, BT+FC or BT+FC+AC-3).
4. **Feedback loop.** If there is no solution, retry with k = 5 routes (skipped when the route lists do not change), then drop
   the lowest-priority request ("wavelengths exhausted", or "solver timeout" if the solve hit the 1,000,000-assignment cap) and
   solve again until a solution exists. Every solve is logged.
5. **Strict mode** never drops anything and returns the CSP's message, e.g. `No solution: wavelengths exhausted on link X-Y`.

Every returned assignment is re-checked with the validator `is_valid`. `evaluate(result, true_demand)` computes the metrics used
in Experiment 3 on the real traffic: **served share** (true demand of the true top-K pairs that were selected and assigned,
divided by the true demand of all true top-K pairs), **blocking rate** (blocked / K) and **top-K hit rate**.

**Findings.**
- On Abilene with a 5,000 km reach, retrying with 5 routes never fixed a failed 3-route solve (0 of 2,688 real instances scanned:
  K from 5 to 40, W from 2 to 8, 56 test hours, strict mode). The topology has only 15 links. The "more routes fix a failure" test
  therefore uses a toy network with five link-disjoint routes. The case where a request must be dropped is real (NN forecast, hour
  540, K=20, W=4: 8 of 20 requests served).
- The brief's drop rule (always drop the globally lowest-priority request) is conservative. With the static highest-demand-first
  order, the unsolvable core is often among the top requests, so dropping low-priority ones repeats an identical failing search
  until the fitting prefix is left (13 failing solves with identical counts in the example above). It is implemented as the
  brief describes; dropping the lowest-priority request involved in the conflict is a possible variant and was not run.

## 4. Experiment 3: does the NN help the planner?

Design: 168 test hours x 5 NN seeds, K = 20, W = 4 and W = 8, planner BT+FC+AC-3 with precheck. The same instances are planned
with the NN forecast and with the last-hour forecast. The interval is a moving-block bootstrap (24-hour blocks, 2,000 resamples,
seed 42) on the paired per-hour difference NN minus last-hour; only 7 blocks exist, so the intervals are rough.

| | metric | last-hour | NN (mean ± std over 5 seeds) | NN minus last-hour, paired | 95% block-bootstrap CI |
|---|---|---|---|---|---|
| W=4 | served share | 0.598 | 0.597 ± 0.002 | -0.0006 | [-0.0075, +0.0108] |
| W=4 | blocking rate | 0.581 | 0.587 ± 0.002 | +0.0059 | [+0.0013, +0.0093] |
| W=4 | top-K hit rate | 0.898 | 0.901 ± 0.001 | +0.0029 | [-0.0026, +0.0085] |
| W=8 | served share | 0.943 | 0.944 ± 0.002 | +0.0011 | [-0.0023, +0.0044] |
| W=8 | blocking rate | 0.000 | 0.000 | 0 | 0 |

**Reading.** The NN gives no practical gain over the last-hour forecast. The one interval that excludes zero goes against the NN:
at W=4 it blocks about 0.6 percentage points more requests. The top-K hit rate does not depend on W, so its two rows are one
comparison, and blocking at W=8 is zero for every run, which leaves four distinct comparisons; a borderline interval should be
read with caution. Figures: `outputs/experiments/exp3_nn.png`, `outputs/demo_cache/demo_nn_effect.png`. Raw rows:
`outputs/experiments/exp3_nn_raw.csv`.

## 5. Limits that concern Member A's part

- Observational 2004 data reused as if it were a recent network; no traffic destined for the Abilene AS.
- Hourly forecasts: the network is planned for the hourly mean, not for 5-minute peaks.
- One lightpath per request; every hourly demand fits in one lightpath (largest 5.2 Gbps against 9,920 Mbps, asserted by a test).
- Four weeks of data and one test week: the Experiment 3 intervals rest on 168 correlated hours (7 daily blocks).
- The conclusion "the NN does not help" applies to this NN and this series.
- Only Abilene was used for the full pipeline; the reach limit (5,000 km) and retry route count (5) are chosen values.
- Everything is simulated; no real equipment and no physical-layer model beyond a fixed reach limit.

Citations: S. Orlowski, R. Wessaly, M. Pioro, A. Tomaszewski, "SNDlib 1.0 - Survivable Network Design Library", Networks 55(3),
2010; Yin Zhang's Abilene traffic-matrix dataset.
