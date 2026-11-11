# Data dictionary (Step 1.12)

## Real data
| Item | Source | Description |
|---|---|---|
| Traffic demands | Yin Zhang, Abilene TMs, weeks X06-X09 (2004-05-01 to 2004-05-28) | Real measured block (block 0 of 5) of 12x12 matrices, 5-min resolution, unit 100 bytes/5 min. Observational data, no traffic destined for the Abilene AS (AS11537). |
| Routing matrix | `A.txt` (Zhang) | Link/demand incidence with fractions; used only for the mapping check (Step 1.9). |
| Topology | SNDlib Abilene (12 nodes, 15 links, lon/lat), germany50 (50 nodes, 88 links) | Copies from public mirrors (see README). |

## Derived data
| Item | How it is derived |
|---|---|
| Mbps rates | v x 100 x 8 / (300 x 1e6) (verified against the SNDlib sample matrix, ratio 2.67e-6). |
| Hourly demand | Mean of 12 consecutive 5-min Mbps values; array (672, 132). |
| 132 ordered pairs | 12x12 matrix, rows = sender, cols = receiver, diagonal removed, src-major, alphabetical router order. |
| Splits | Chronological: train hours 0-503, validation 432-503 (last 3 days of train), test 504-671. First usable target hour 168 (lag-168 feature). |
| Link lengths (km) | Haversine distance between endpoint coordinates (Member B, Step 1.7b). |

## Generated parameters (all in `config.yaml`)
| Parameter | Value | How it is set |
|---|---|---|
| Seed | 42 | Fixed global seed; every random choice uses it or a documented derivative. |
| W (wavelengths) | 4 or 8 | Experiment variable. |
| K (top-K requests) | 10 / 20 / 40 | Experiment variable. |
| Reach limit (km) | 4000 (placeholder) | Maximum length of a lightpath route; final value set by Member B at Step 1.7b from the Abilene link lengths. |
| Routes per request / max routes | 3 / 5 | 3 first, 5 in the feedback loop (Step 5.4). |
| Lightpath capacity (Mbps) | 9920 | OC-192 usable rate, equal to the capacity of 14 of the 15 SNDlib Abilene links (one link has 2480). One lightpath per request; every request fits, since the largest hourly pair demand observed is about 5.2 Gbps. |
| CSP safety cap | 1,000,000 assignments | Returns status `timeout` with the counters. |
