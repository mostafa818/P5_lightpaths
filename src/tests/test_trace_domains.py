"""Stage 6 tests (Member B): search trace, domain-size analysis, plot (c)."""
import pytest

from common import Request, Route, Stats
from csp.analysis import domain_sizes
from search.algorithms import astar, ucs
from search.graph import Graph
from search.trace import expansion_trace, format_trace


def route(length, *links):
    return Route(nodes=("a", "b"), links=frozenset(tuple(sorted(l)) for l in links), length_km=length)


def req(i, d=1.0):
    return Request(i, "a", "b", d)


# ---------------- trace ----------------
@pytest.fixture(scope="module")
def abilene_graph():
    from eval.exp1_search import graphs
    try:
        return graphs()["Abilene"]
    except FileNotFoundError:
        pytest.skip("raw topology files not downloaded")


def test_trace_matches_the_algorithms_and_their_counters(abilene_graph):
    g = abilene_graph
    for algo, fn in (("ucs", ucs), ("astar", astar)):
        t = expansion_trace(g, "KSCYng", "LOSAng", algo)
        st = Stats()
        path, cost = fn(g, "KSCYng", "LOSAng", st)
        assert t["path"] == path and t["cost_km"] == cost
        assert (t["expanded"], t["generated"]) == (st.expanded, st.generated)
        assert len(t["order"]) == t["expanded"] and len(set(t["order"])) == t["expanded"]
        assert t["order"][0] == "KSCYng" and t["order"][-1] == "LOSAng"


def test_astar_expands_fewer_nodes_for_the_demo_pair_and_same_cost(abilene_graph):
    u = expansion_trace(abilene_graph, "KSCYng", "LOSAng", "ucs")
    a = expansion_trace(abilene_graph, "KSCYng", "LOSAng", "astar")
    assert (u["expanded"], a["expanded"]) == (12, 4)
    assert a["cost_km"] == pytest.approx(u["cost_km"])


def test_trace_edge_cases_and_format():
    g = Graph.from_edges([("A", "B", 1.0), ("C", "D", 1.0)])
    with pytest.raises(ValueError):
        expansion_trace(g, "A", "B", "dfs")
    t = expansion_trace(g, "A", "D", "ucs")                  # unreachable
    assert t["path"] == [] and t["cost_km"] is None and t["order"] == ["A", "B"]
    u, a = expansion_trace(g, "A", "B", "ucs"), expansion_trace(g, "A", "B", "ucs")
    text = format_trace(u, a)
    assert "A *" in text and "B *" in text and "expanded 2" in text and "0 fewer nodes" in text


# ---------------- domain sizes ----------------
def test_domain_sizes_hand_example_forward_checking():
    # two requests share link X-Y on their only route, W = 2: 2 values each; assigning r1 = (route 0, w0)
    # removes w0 from r2
    rs = [req(1, 5), req(2, 1)]
    routes = {1: [route(10, "XY")], 2: [route(10, "XY")]}
    d = domain_sizes(rs, routes, 2)
    assert d["requests"] == [1, 2]
    assert d["initial"] == [2, 2] and d["after_ac3"] == [2, 2] and d["after_fc"] == [1, 1]
    assert not d["ac3_wiped_out"] and not d["fc_wiped_out"]


def test_domain_sizes_priority_order_and_disjoint_requests_are_untouched():
    rs = [req(1, 1), req(2, 9)]
    routes = {1: [route(10, "AB")], 2: [route(10, "CD"), route(12, "EF")]}
    d = domain_sizes(rs, routes, 2)
    assert d["requests"] == [2, 1] and d["initial"] == [4, 2]
    assert d["after_fc"] == [1, 2]                           # request 1 is not a neighbour: unchanged


def test_domain_sizes_ac3_wipe_out_at_w1():
    rs = [req(i, 10 - i) for i in (1, 2)]
    routes = {1: [route(10, "XY")], 2: [route(10, "XY")]}
    d = domain_sizes(rs, routes, 1)
    assert d["ac3_wiped_out"] and d["after_fc"] is None
    assert d["wiped_out_request"] in (1, 2) and 0 in d["after_ac3"]


def test_domain_sizes_fc_steps_and_fc_wipe_out():
    rs = [req(i, 10 - i) for i in (1, 2, 3)]
    routes = {i: [route(10, "XY")] for i in (1, 2, 3)}       # W = 2: third request has no value left
    d1, d2 = domain_sizes(rs, routes, 2, fc_steps=1), domain_sizes(rs, routes, 2, fc_steps=2)
    assert d1["after_fc"] == [1, 1, 1] and not d1["fc_wiped_out"]
    assert d2["fc_wiped_out"] and d2["fc_wiped_request"] == 3 and d2["after_fc"] == [1, 1, 0]


def test_domain_sizes_empty_and_ac3_unchanged_at_w2_on_real_instance():
    assert domain_sizes([], {}, 2)["after_fc"] == []
    from eval import exp2_csp as e2
    try:
        planner, H, _ = e2.default_planner()
    except FileNotFoundError:
        pytest.skip("data not available")
    reqs, routes = e2.instance(planner, 540, 20, H[540])
    d = domain_sizes(reqs, routes, 4, fc_steps=3)
    assert d["after_ac3"] == d["initial"]                    # AC-3 prunes nothing at W >= 2 (Experiment 2)
    assert sum(d["after_fc"]) < sum(d["initial"])


def test_make_plot_writes_a_png(tmp_path):
    from eval.plot_csp import make_plot
    rs = [req(1, 5), req(2, 1)]
    routes = {1: [route(10, "XY")], 2: [route(10, "XY")]}
    out = make_plot([("toy W=2", domain_sizes(rs, routes, 2)), ("toy W=1", domain_sizes(rs, routes, 1))],
                    tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 5000 and out.endswith("p.png")
    make_plot(domain_sizes(rs, routes, 2), tmp_path / "single.png")
