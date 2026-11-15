"""Smoke tests for the experiment code (small slices only; the full runs are `python -m eval.run_experiments`)."""
import numpy as np
import pandas as pd
import pytest

from csp.solver import solve
from eval import exp1_search as e1
from eval import exp2_csp as e2


@pytest.fixture(scope="module")
def graphs():
    try:
        return e1.graphs()
    except FileNotFoundError:
        pytest.skip("raw topology files not downloaded")


def test_exp1_counts_are_deterministic_and_astar_is_never_worse_in_cost(graphs):
    g = graphs["Abilene"]
    pairs = [("ATLAM5", "WASHng"), ("LOSAng", "NYCMng"), ("STTLng", "HSTNng")]
    a, b = pd.DataFrame(e1.one_run(g, pairs)), pd.DataFrame(e1.one_run(g, pairs))
    cols = ["expanded", "generated", "cost_km", "hops"]
    assert a[cols].equals(b[cols])
    for src, dst in pairs:
        d = a[(a.src == src) & (a.dst == dst)].set_index("algo")
        assert d.cost_km["A*"] == pytest.approx(d.cost_km["UCS"])
        assert d.expanded["A*"] <= d.expanded["UCS"]
        assert d.cost_km["BFS"] >= d.cost_km["UCS"] - 1e-9


def test_exp1_sampling_is_seeded_and_bfs_has_fewest_hops(graphs):
    g = graphs["germany50"]
    assert e1.sample_pairs(g, 42) == e1.sample_pairs(g, 42) != e1.sample_pairs(g, 43)
    assert len(set(e1.sample_pairs(g, 42))) == e1.N_RANDOM
    raw = pd.DataFrame(e1.one_run(g, e1.sample_pairs(g, 42)[:10]))
    raw.insert(0, "graph", "germany50")
    assert e1.check_bfs_fewest_hops(raw, {"germany50": g}) == 0


def test_exp2_methods_agree_on_a_real_instance():
    try:
        planner, H, _ = e2.default_planner()
    except FileNotFoundError:
        pytest.skip("raw topology files not downloaded")
    t = e2.HOURS[0]
    reqs, routes = e2.instance(planner, t, 10, H[t])
    out = {n: solve(reqs, routes, 8, use_fc=f, use_ac3=a, max_assignments=200_000)
           for n, (f, a) in e2.METHODS.items()}
    assert {r.status for r in out.values()} == {"ok"}
    assert out["BT+FC"].assignment == out["BT"].assignment == out["BT+FC+AC3"].assignment
    assert out["BT+FC"].stats.assignments <= out["BT"].stats.assignments


def test_exp2_hours_lie_in_the_test_week_and_are_distinct():
    assert len(set(e2.HOURS)) == 10 and all(504 <= h <= 671 for h in e2.HOURS)
