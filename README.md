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
