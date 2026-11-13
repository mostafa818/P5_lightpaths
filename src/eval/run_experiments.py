"""Regenerate every experiment table and plot (plan 7.4):  python -m eval.run_experiments [exp1 exp2 ...]

Outputs go to outputs/experiments/: raw CSVs, ``*_summary.md`` tables and PNG plots. Experiment 3 (Member A)
registers itself in EXPERIMENTS below.
"""
from __future__ import annotations

import importlib
import sys

EXPERIMENTS = {"exp1": "eval.exp1_search", "exp2": "eval.exp2_csp"}     # exp3: A adds "eval.exp3_nn"


def main(argv: list) -> None:
    chosen = argv or list(EXPERIMENTS)
    unknown = [c for c in chosen if c not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s) {unknown}; choose from {sorted(EXPERIMENTS)}")
    for name in chosen:
        print(f"=== {name} ===", flush=True)
        importlib.import_module(EXPERIMENTS[name]).main()


if __name__ == "__main__":
    main(sys.argv[1:])
