# Data dictionary (Steps 1.12 and 9.2)

Three kinds of items: **real** (measured data, never edited), **derived** (computed from real data by a documented
rule) and **generated** (chosen or simulated by us, all in `config.yaml`). The limitations that follow from this are listed
in the README under [Limitations](../README.md#limitations).

## Real data
| Item | Source | Description |
|---|---|---|
| Traffic demands | Yin Zhang, Abilene traffic matrices, weeks X06 to X09 (2004-05-01 to 2004-05-28) | The real measured block (block 0 of 5) of 12 x 12 matrices at 5-minute resolution, unit 100 bytes per 5 minutes. Observational data; it does not include traffic destined for the Abilene network itself (AS11537). The other four blocks are gravity-model estimates and are not used. |
| Routing matrix | `A.txt` (Zhang) | Link/demand incidence with fractions. Used only for the node-mapping check (Step 1.9). |
| Abilene topology | SNDlib `abilene_sndlib_native.txt` | 12 nodes, 15 links, longitude and latitude per node, link capacities (14 links 9,920 Mbps, one 2,480 Mbps). A copy from a public mirror of SNDlib. |
| germany50 topology | TopoHub copy of the SNDlib network | 50 nodes, 88 links, node positions and `dist` in km. Used only in Experiment 1. |
| SNDlib sample matrix | `demandMatrix-abilene-zhang-5min-20040301-0145.txt` | Same Zhang data under SNDlib node names; used only to verify the node mapping. |

## Derived data
| Item | How it is derived |
|---|---|
| Mbps rates | value x 100 x 8 / (300 x 1e6), verified against the SNDlib sample matrix (ratio 2.67e-6). |
| Hourly demand | Mean of 12 consecutive 5-minute Mbps values. Array of shape (672, 132) in `data/processed/hourly.npz`. |
| 132 ordered pairs | Rows are senders, columns receivers, diagonal removed, sender-major, alphabetical router order. The list is stored in `data/processed/meta.json`; a request's `id` is its index in this list. |
| Splits | Chronological: train hours 0-503, validation 432-503 (the last 3 days of train), test 504-671. The first usable target hour is 168 (lag-168 feature). Hour 0 is 2004-05-01 00:00, a Saturday. |
| Link lengths (km) | Haversine distance between the endpoint longitude and latitude of each of the 15 SNDlib Abilene links: 132 km to 2,193 km, 14,029 km in total (Member B, Step 1.7b). Saved in `data/processed/abilene_links.csv` by `python -m search.topology`. |
| Requests | For a target hour, the K pairs with the highest *predicted* demand; the prediction is the request's priority. Built by the planner, not stored. |
| Features and scaling | 29 inputs per (hour, pair): demand at t-24 to t-1, demand at t-168, sine and cosine of hour of day and weekday. `log1p`, then standardised with the mean and standard deviation of the training hours (0-431). Saved in `outputs/nn/scaler.json`. |

## Generated parameters (all in `config.yaml`)
"Used by" says which code reads the value, so no listed parameter is decorative.

| Parameter | Value | How it is set | Used by |
|---|---|---|---|
| Seed | 42 | Fixed global seed; every random choice uses it or a documented derivative (the NN seeds 0-4 are separate and listed below). | `make_data.py`, experiments |
| W (wavelengths) | default 8 | Default of `run_demo.py`. An experiment variable: Experiments 2 and 3 use W = 4 and W = 8 (set inside the experiment scripts). | `run_demo.py` |
| K (top-K requests) | default 20 | Default of `run_demo.py`. An experiment variable: Experiment 2 uses K = 10, 20, 40, Experiment 3 K = 20. | `run_demo.py` |
| Reach limit (km) | 5000 | Maximum length of a lightpath route. Set at Step 1.7b from the link lengths: the longest shortest path is 4,706 km (WASHng-STTLng), so 4000 would leave 12 of the 132 pairs without any route. At 5000 km every pair has a route, 120 pairs have at least 2 and 98 have 3. A chosen value, not a measured physical limit. | search, integration |
| Routes per request / max routes | 3 / 5 | 3 on the first attempt, 5 in the retry of the feedback loop (Step 5.4). | integration |
| Lightpath capacity (Mbps) | 9920 | The OC-192 rate, equal to the capacity of 14 of the 15 Abilene links. It is **not** a constraint in the planner: one lightpath carries a whole request. It documents that assumption, and a test asserts that every hourly demand fits (the largest is 5,196 Mbps). | `tests` |
| CSP safety cap | 1,000,000 assignments | Per solve; a solve that reaches it returns status `timeout` with the counters. | csp, integration, experiments |
| NN settings | seeds 0-4, hidden [64, 64], lr 0.001, batch 256, up to 100 epochs, patience 10 | Chosen once, not tuned (no tuning on the test set); early stopping on the chronological validation MSE on the log scale. | `nn/train.py` |
| Split hours | train 0-503, test 504-671, validation last 3 days | Chronological, to avoid leaking the future; see Derived data. | `make_data.py`, `nn` |
| Node order | ATLAM5, ATLAng, CHINng, DNVRng, HSTNng, IPLSng, KSCYng, LOSAng, NYCMng, SNVAng, STTLng, WASHng | Alphabetical router names; verified in Step 1.9 (the `A.txt` order, and a Spearman correlation of 1.000 against the SNDlib sample). | `make_data.py`, integration |
| Experiment test hours | 504 + 17 i, i = 0..9 (Experiment 2); all 168 test hours (Experiment 3) | Spread over the test week so different hours of the day and weekdays appear. | experiments |
| Bootstrap | 24-hour blocks, 2,000 resamples, seed 42 | Moving-block bootstrap for the Experiment 3 intervals (hourly traffic is autocorrelated). | `eval/exp3_nn.py` |
