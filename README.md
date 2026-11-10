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
python -m eval.run_experiments
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

## Layout
`config.yaml` (all parameters) | `src/{data,nn,search,csp,integration,eval}/` | `src/common.py` (interface contract) | `src/tests/` | `data/{raw,processed}/` | `outputs/` | `report/` | `docs/`

## Counter definitions (both members MUST count this way)
**Search**
- *nodes generated*: every node pushed onto the frontier.
- *nodes expanded*: every node popped from the frontier and processed.

**CSP**
- *assignments*: every tentative variable = value attempt.
- *backtracks*: every time a variable has no value left and the search returns.

## Git workflow
Branches `member-a/...` and `member-b/...`; merge via pull request reviewed by the other member; small, frequent commits.

## Limitations
(to be completed in Stage 9)
