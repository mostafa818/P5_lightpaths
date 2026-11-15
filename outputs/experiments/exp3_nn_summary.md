## Experiment 3: NN vs last-hour forecast inside the planner

168 test hours (504-671), 5 NN seeds, planner BT+FC+AC-3 with precheck, cap 1,000,000 assignments, k = 3 routes (5 on retry), reach 5000 km. Served share and top-K hit rate: higher is better; blocking rate: lower is better. The last-hour forecast is deterministic, so it has no seed spread. The paired difference averages the 5 seeds per hour; its s.e. treats the 168 hours as independent, which they are not (hourly traffic is autocorrelated), so the moving-block bootstrap CI (24-hour blocks, 2000 resamples, seed 42) is the more reliable interval; with only 7 blocks it is still rough. The top-K hit rate depends on the forecast only, not on W, so its two rows are one comparison; blocking at W=8 is zero for every run. That leaves four distinct comparisons (served share at W=4 and W=8, blocking at W=4, hit rate), so a borderline interval should be read with caution.

### K=20, W=4

| metric | last-hour | NN (mean ± std over 5 seeds) | NN − last-hour, paired (mean ± s.e. over hours) | 95% block-bootstrap CI | hours NN better / equal / worse |
|---|---|---|---|---|---|
| served share | 0.598 | 0.597 ± 0.002 | -0.0006 ± 0.0033 | [-0.0075, +0.0108] | 51 / 44 / 73 |
| blocking rate | 0.581 | 0.587 ± 0.002 | +0.0059 ± 0.0022 | [+0.0013, +0.0093] | 37 / 73 / 58 |
| top-K hit rate | 0.898 | 0.901 ± 0.001 | +0.0029 ± 0.0024 | [-0.0026, +0.0085] | 69 / 42 / 57 |

Per-seed paired mean difference NN − last-hour (served share: +0.0002, -0.0025, +0.0008, -0.0027, +0.0015; blocking rate: +0.0060, +0.0062, +0.0054, +0.0080, +0.0039; top-K hit rate: +0.0030, +0.0018, +0.0027, +0.0033, +0.0036).

### K=20, W=8

| metric | last-hour | NN (mean ± std over 5 seeds) | NN − last-hour, paired (mean ± s.e. over hours) | 95% block-bootstrap CI | hours NN better / equal / worse |
|---|---|---|---|---|---|
| served share | 0.943 | 0.944 ± 0.002 | +0.0011 ± 0.0021 | [-0.0023, +0.0044] | 76 / 28 / 64 |
| blocking rate | 0.000 | 0.000 ± 0.000 | +0.0000 ± 0.0000 | [+0.0000, +0.0000] | 0 / 168 / 0 |
| top-K hit rate | 0.898 | 0.901 ± 0.001 | +0.0029 ± 0.0024 | [-0.0026, +0.0085] | 69 / 42 / 57 |

Per-seed paired mean difference NN − last-hour (served share: +0.0034, +0.0016, -0.0024, -0.0003, +0.0032; blocking rate: +0.0000, +0.0000, +0.0000, +0.0000, +0.0000; top-K hit rate: +0.0030, +0.0018, +0.0027, +0.0033, +0.0036).
