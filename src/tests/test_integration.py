"""Stage 5 tests (plan 5.7, 5.8, 8.1): toy network and Abilene; every output is re-validated by an
independent checker (no shared link on the same wavelength) as well as csp.model.is_valid."""
import itertools, os
import numpy as np, pytest

from csp.model import is_valid
from integration.pipeline import (Planner, plan, evaluate, select_requests, NO_ROUTE, EXHAUSTED, TIMEOUT)
from search.graph import Graph

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOURLY = os.path.join(ROOT, "data", "processed", "hourly.npz")
MODEL = os.path.join(ROOT, "outputs", "nn", "mlp_seed0.joblib")
TOPO = os.path.join(ROOT, "data", "raw", "topology", "abilene_sndlib_native.txt")
needs_abilene = pytest.mark.skipif(not (os.path.exists(HOURLY) and os.path.exists(MODEL) and os.path.exists(TOPO)),
                                   reason="needs processed data, NN models and the Abilene topology file")


def independent_check(result, W):
    """True if no two served requests share a link on the same wavelength (own implementation)."""
    served = result.lightpaths
    for a, b in itertools.combinations(served, 2):
        if a["wavelength"] == b["wavelength"]:
            la = set(zip(a["route"], a["route"][1:])) | set(zip(a["route"][1:], a["route"]))
            lb = set(zip(b["route"], b["route"][1:]))
            if la & lb:
                return False
    return all(0 <= s["wavelength"] < W for s in served)


# ---------- toy networks ----------------------------------------------------------------------------
def five_route_planner(reach=100.0):
    """S and D joined by five link-disjoint two-hop routes of increasing length."""
    mids = ["A", "B", "C", "E", "F"]
    edges = []
    for i, m in enumerate(mids):
        edges += [("S", m, 1.0 + 0.1 * i), (m, "D", 1.0 + 0.1 * i)]
    return Planner(Graph.from_edges(edges), [("S", "D")] * 5, reach, routes_first=3, routes_max=5)


def test_evidence_more_routes_fix_a_failure():
    """Plan 5.7 (named case): 5 requests S->D, W=1 need 5 disjoint routes; k=3 cannot, k=5 can."""
    pl = five_route_planner()
    predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    r = plan(pl, 0, predicted, K=5, W=1)
    assert [it["k"] for it in r.iterations] == [3, 5]
    assert r.iterations[0]["status"] == "no_solution" and r.iterations[1]["status"] == "ok"
    assert r.status == "ok" and not r.blocked and len(r.assignments) == 5
    assert independent_check(r, 1)


def test_evidence_request_must_be_dropped_on_toy():
    """Plan 5.7 (named case): 6 requests but only 5 disjoint routes within reach and W=1 -> one is dropped."""
    mids = ["A", "B", "C", "E", "F"]
    edges = []
    for i, m in enumerate(mids):
        edges += [("S", m, 1.0 + 0.1 * i), (m, "D", 1.0 + 0.1 * i)]
    pl = Planner(Graph.from_edges(edges), [("S", "D")] * 6, 100.0)
    r = plan(pl, 0, np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), K=6, W=1)
    assert [(b.id, why) for b, why in r.blocked] == [(5, EXHAUSTED)]       # lowest predicted demand dropped
    assert len(r.assignments) == 5 and independent_check(r, 1)
    assert [it["k"] for it in r.iterations][:2] == [3, 5]                  # more routes were tried first


def single_route_planner(n=3, reach=100.0):
    return Planner(Graph.from_edges([("X", "Y", 10.0)]), [("X", "Y")] * n, reach)


def test_impossible_case_a_strict_and_normal():
    """Three requests whose only route shares one link, W=2 (plan 3.9a at integration level)."""
    pl, pred = single_route_planner(3), np.array([3.0, 2.0, 1.0])
    s = plan(pl, 0, pred, K=3, W=2, strict=True)
    assert s.status == "no_solution" and s.reason == "No solution: wavelengths exhausted on link X-Y"
    assert not s.assignments and not s.blocked
    n = plan(pl, 0, pred, K=3, W=2)
    assert n.status == "ok" and sorted(n.assignments) == [0, 1]
    assert [(b.id, why) for b, why in n.blocked] == [(2, EXHAUSTED)]
    assert independent_check(n, 2)


def test_impossible_case_b_no_route_within_reach():
    pl = Planner(Graph.from_edges([("X", "Y", 10.0), ("Y", "Z", 500.0)]), [("X", "Y"), ("X", "Z")], reach_km=100.0)
    pred = np.array([1.0, 2.0])                                           # X->Z (id 1) is out of reach
    n = plan(pl, 0, pred, K=2, W=2)
    assert n.status == "ok" and [(b.id, why) for b, why in n.blocked] == [(1, NO_ROUTE)]
    assert n.iterations[0]["n_requests"] == 1                              # never entered the CSP
    s = plan(pl, 0, pred, K=2, W=2, strict=True)
    assert s.status == "no_solution" and s.reason == "No solution: no route within reach for request 1"


def test_drop_order_lowest_demand_then_larger_id():
    pl = single_route_planner(4)
    r = plan(pl, 0, np.array([5.0, 2.0, 2.0, 9.0]), K=4, W=2)             # keep 2: ids 3 and 0; drop 1,2
    assert sorted(r.assignments) == [0, 3]
    assert [b.id for b, _ in r.blocked] == [2, 1]                          # tie on 2.0 -> larger id dropped first


def test_solver_timeout_is_a_failure_not_a_crash():
    pl = single_route_planner(3)
    r = plan(pl, 0, np.array([3.0, 2.0, 1.0]), K=3, W=3, max_assignments=1)
    assert r.status == "ok" and len(r.assignments) == 1
    assert {why for _, why in r.blocked} == {TIMEOUT}


# ---------- edge cases (plan 8.1) -------------------------------------------------------------------
def test_edge_k_equals_one():
    r = plan(single_route_planner(3), 0, np.array([1.0, 9.0, 2.0]), K=1, W=1)
    assert [x.id for x in r.requests] == [1] and sorted(r.assignments) == [1] and not r.blocked


def test_edge_w1_disjoint_requests_share_wavelength():
    g = Graph.from_edges([("A", "B", 5.0), ("C", "D", 5.0)])
    pl = Planner(g, [("A", "B"), ("C", "D")], 100.0)
    r = plan(pl, 0, np.array([2.0, 1.0]), K=2, W=1)
    assert len(r.assignments) == 2 and {w for _, w in r.assignments.values()} == {0}


def test_edge_reach_exactly_equal_to_route_length():
    g = Graph.from_edges([("A", "B", 7.5)])
    r = plan(Planner(g, [("A", "B")], reach_km=7.5), 0, np.array([1.0]), K=1, W=1)
    assert len(r.assignments) == 1 and not r.blocked
    r2 = plan(Planner(g, [("A", "B")], reach_km=7.4), 0, np.array([1.0]), K=1, W=1)
    assert [why for _, why in r2.blocked] == [NO_ROUTE]


def test_edge_tied_demands_are_deterministic():
    pairs = [("S", "D")] * 5
    sel = select_requests(np.array([3.0, 3.0, 3.0, 1.0, 3.0]), pairs, 3)
    assert [r.id for r in sel] == [0, 1, 2]                                # ties: lower pair index first
    pl = five_route_planner()
    a = plan(pl, 0, np.array([3.0, 3.0, 3.0, 3.0, 3.0]), K=5, W=1)
    b = plan(pl, 0, np.array([3.0, 3.0, 3.0, 3.0, 3.0]), K=5, W=1)
    assert a.assignments == b.assignments


def test_methods_agree_and_unknown_method_rejected():
    pl = five_route_planner()
    pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    outs = [plan(pl, 0, pred, K=5, W=1, method=m) for m in ("bt", "fc", "fc+ac3")]
    assert outs[0].assignments == outs[1].assignments == outs[2].assignments
    with pytest.raises(ValueError):
        plan(pl, 0, pred, K=5, W=1, method="nope")


def test_metrics_on_toy():
    pl = single_route_planner(3)
    r = plan(pl, 0, np.array([3.0, 2.0, 1.0]), K=3, W=2)                  # serves ids 0,1; blocks id 2
    m = evaluate(r, np.array([30.0, 20.0, 10.0]))
    assert m["served_share"] == pytest.approx(50 / 60) and m["blocking_rate"] == pytest.approx(1 / 3)
    assert m["topk_hit_rate"] == 1.0
    s = plan(pl, 0, np.array([3.0, 2.0, 1.0]), K=3, W=2, strict=True)
    ms = evaluate(s, np.array([30.0, 20.0, 10.0]))
    assert ms["served_share"] == 0.0 and ms["blocking_rate"] == 1.0
    # a wrong forecast: selected {0,1} but true top-2 is {1,2} -> only id 1 is both selected and in the true top-K
    r2 = plan(pl, 0, np.array([3.0, 2.0, 0.0]), K=2, W=2)
    m2 = evaluate(r2, np.array([1.0, 20.0, 30.0]))
    assert m2["served_share"] == pytest.approx(20 / 50) and m2["topk_hit_rate"] == 0.5


# ---------- Abilene -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def abilene():
    from integration.pipeline import default_planner
    pl, H, cfg = default_planner()
    return pl, H, cfg


@needs_abilene
@pytest.mark.parametrize("method", ["bt", "fc", "fc+ac3"])
@pytest.mark.parametrize("K,W", [(10, 4), (10, 8), (20, 8)])
def test_abilene_full_pipeline_outputs_are_valid(abilene, method, K, W):
    from nn.forecast import forecast
    pl, H, cfg = abilene
    for t in (504, 540, 600):
        r = plan(pl, t, forecast(t), K=K, W=W, method=method, max_assignments=200000)
        assert r.status == "ok" and independent_check(r, W)
        active = [x for x in r.requests if x.id in r.assignments]
        assert is_valid(active, r.assignments, W, r.routes_by_request, pl.reach_km, require_all=True)
        assert len(r.assignments) + len(r.blocked) == K
        assert all(s["length_km"] <= pl.reach_km for s in r.lightpaths)


@needs_abilene
def test_evidence_request_must_be_dropped_on_abilene(abilene):
    """Plan 5.7 (named real case): NN forecast, hour 540, K=20, W=4: not solvable, requests are dropped."""
    from nn.forecast import forecast
    pl, H, cfg = abilene
    r = plan(pl, 540, forecast(540), K=20, W=4)
    assert r.iterations[0]["status"] == "no_solution" and r.iterations[1]["k"] == 5   # one request has a 4th route
    assert len(r.blocked) > 0 and {why for _, why in r.blocked} == {EXHAUSTED}
    assert len(r.assignments) + len(r.blocked) == 20 and independent_check(r, 4)
    m = evaluate(r, H[540])
    assert 0 < m["served_share"] < 1 and m["blocking_rate"] == len(r.blocked) / 20
    assert plan(pl, 540, forecast(540), K=20, W=4, strict=True).status == "no_solution"


@needs_abilene
def test_abilene_no_future_data_reaches_the_planner(abilene):
    from nn import forecast as FC
    pl, H, cfg = abilene
    FC._load(0)
    saved = FC._cache["H"]
    try:
        a = plan(pl, 520, FC.forecast(520), K=10, W=8)
        bad = saved.copy(); bad[520:] = 7777.0
        FC._cache["H"] = bad
        b = plan(pl, 520, FC.forecast(520), K=10, W=8)
        assert a.assignments == b.assignments and [x.id for x in a.requests] == [x.id for x in b.requests]
    finally:
        FC._cache["H"] = saved


def test_more_routes_retry_skipped_when_route_lists_unchanged():
    r = plan(single_route_planner(3), 0, np.array([3.0, 2.0, 1.0]), K=3, W=2)    # only one route per request exists
    assert [it["note"] for it in r.iterations][0] == "first attempt"
    assert all(it["note"] != "more routes" for it in r.iterations)
    assert r.iterations[0]["status"] == "no_solution" and len(r.iterations) == 2     # first attempt + one after the drop


def test_strict_timeout_keeps_its_own_status():
    s = plan(single_route_planner(3), 0, np.array([3.0, 2.0, 1.0]), K=3, W=3, strict=True, max_assignments=1)
    assert s.status == "timeout" and "Timeout" in s.reason and not s.assignments
    m = evaluate(s, np.array([3.0, 2.0, 1.0]))
    assert m["served_share"] == 0.0 and m["blocking_rate"] == 1.0
