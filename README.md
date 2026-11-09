# P5 Lightpaths: Route and Wavelength Assignment (Search + CSP + NN)

## Setup
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
