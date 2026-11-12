"""Up to k loop-free routes with Yen's algorithm (plan Step 2.8) and the reach limit."""
import itertools
import random

import networkx as nx
import pytest

import common
from common import Route, Stats, link_id
from search.routes import k_routes, k_shortest_paths
from tests.conftest import to_networkx

TOY_AE = [(["A", "B", "D", "E"], 9), (["A", "B", "C", "D", "E"], 12), (["A", "C", "D", "E"], 14),
          (["A", "B", "D", "C", "E"], 15), (["A", "B", "C", "E"], 16), (["A", "C", "E"], 18),
          (["A", "C", "B", "D", "E"], 19)]            # all 7 simple A->E paths, worked out by hand


def check_routes(graph, routes, src, dst):
    lengths = [r[1] for r in routes]
    assert lengths == sorted(lengths)                              # shortest first
    assert len({tuple(r[0]) for r in routes}) == len(routes)       # distinct
    for nodes, length in routes:
        assert nodes[0] == src and nodes[-1] == dst
        assert len(set(nodes)) == len(nodes)                       # loop-free
        assert length == pytest.approx(graph.path_length(nodes))   # every consecutive pair is a link


# ---- hand-computed toy graph --------------------------------------------------------------------------------
def test_all_seven_simple_paths_in_order(toy):
    got = k_shortest_paths(toy, "A", "E", 20)
    assert [(n, l) for n, l in got] == TOY_AE


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7])
def test_k_limits_the_number_of_routes(toy, k):
    assert k_shortest_paths(toy, "A", "E", k) == [(n, l) for n, l in TOY_AE[:k]]


def test_fewer_than_k_when_fewer_exist(toy):
    assert len(k_shortest_paths(toy, "A", "E", 50)) == 7


@pytest.mark.parametrize("reach,expected", [
    (8.99, 0), (9, 1),               # reach exactly equal to the route length keeps it
    (11.99, 1), (12, 2),             # ... for the second route too
    (13, 2), (15, 4), (100, 7)])
def test_reach_limit_is_inclusive(toy, reach, expected):
    got = k_shortest_paths(toy, "A", "E", 20, reach_km=reach)
    assert len(got) == expected
    assert all(length <= reach + 1e-9 for _, length in got)


def test_k_equals_one_returns_the_shortest(toy):
    assert k_shortest_paths(toy, "A", "C", 1) == [(["A", "B", "D", "C"], 7)]


# ---- edge cases -------------------------------------------------------------------------------------------------
def test_src_equals_dst_raises(toy):
    with pytest.raises(ValueError, match="same node"):
        k_routes(toy, "A", "A", 3, 100)


def test_unknown_node_and_bad_k_raise(toy):
    with pytest.raises(ValueError, match="unknown node"):
        k_routes(toy, "A", "Z", 3, 100)
    with pytest.raises(ValueError, match="k must be"):
        k_routes(toy, "A", "E", 0, 100)


def test_unreachable_node_gives_empty_list(toy):
    assert k_routes(toy, "A", "F", 3, 100) == []
    assert k_routes(toy, "F", "A", 3, 100) == []


def test_request_with_no_route_within_reach_gives_empty_list(toy):
    assert k_routes(toy, "A", "E", 3, reach_km=5) == []


# ---- Route objects and the common.py contract --------------------------------------------------------------------
def test_routes_are_route_objects_with_undirected_link_ids(toy):
    routes = k_routes(toy, "E", "A", 2, 100)
    assert all(isinstance(r, Route) for r in routes)
    r = routes[0]
    assert r.nodes == ("E", "D", "B", "A") and r.length_km == pytest.approx(9)
    assert r.links == frozenset({link_id("D", "E"), link_id("B", "D"), link_id("A", "B")})
    assert ("B", "D") in r.links and ("D", "B") not in r.links       # ids are sorted tuples


def test_common_k_routes_delegates(toy):
    assert common.k_routes(toy, "A", "E", 3, 20) == k_routes(toy, "A", "E", 3, 20)


def test_stats_are_collected_across_all_searches(toy):
    st = Stats()
    k_shortest_paths(toy, "A", "E", 3, stats=st)
    assert st.expanded > 0 and st.generated >= st.expanded


# ---- Abilene against networkx.shortest_simple_paths -------------------------------------------------------------
def test_abilene_routes_match_networkx_on_all_pairs(abilene):
    G = to_networkx(abilene)
    for a, b in itertools.permutations(abilene.nodes, 2):
        ref = [nx.path_weight(G, p, "weight")
               for p in itertools.islice(nx.shortest_simple_paths(G, a, b, weight="weight"), 5)]
        got = k_shortest_paths(abilene, a, b, 5)
        check_routes(abilene, got, a, b)
        assert [l for _, l in got] == pytest.approx(ref), (a, b)


def test_abilene_reach_limit_filters_like_networkx(abilene, reach_km):
    G = to_networkx(abilene)
    for a, b in itertools.permutations(abilene.nodes, 2):
        ref = [w for w in (nx.path_weight(G, p, "weight")
                           for p in itertools.islice(nx.shortest_simple_paths(G, a, b, weight="weight"), 3))
               if w <= reach_km]
        assert [l for _, l in k_shortest_paths(abilene, a, b, 3, reach_km)] == pytest.approx(ref)


def test_abilene_route_counts_at_the_configured_reach(abilene, reach_km):
    """With reach_km = 5000 (config.yaml): every pair routable, 120 pairs with >= 2 routes, 98 with 3."""
    assert reach_km == 5000
    counts = [len(k_routes(abilene, a, b, 3, reach_km)) for a, b in itertools.permutations(abilene.nodes, 2)]
    assert len(counts) == 132
    assert sum(c >= 1 for c in counts) == 132
    assert sum(c >= 2 for c in counts) == 120
    assert sum(c >= 3 for c in counts) == 98


def test_atlam5_is_a_leaf_and_all_its_routes_share_one_link(abilene):
    """Only ATLAM5<->ATLAng has a single route; every other ATLAM5 route uses the ATLAM5-ATLAng link."""
    leaf = link_id("ATLAM5", "ATLAng")
    assert len(k_routes(abilene, "ATLAM5", "ATLAng", 5, 5000)) == 1
    assert len(k_routes(abilene, "ATLAng", "ATLAM5", 5, 5000)) == 1
    for other in abilene.nodes:
        if other in ("ATLAM5", "ATLAng"):
            continue
        routes = k_routes(abilene, "ATLAM5", other, 3, 5000) + k_routes(abilene, other, "ATLAM5", 3, 5000)
        assert routes and all(leaf in r.links for r in routes)


# ---- germany50 ------------------------------------------------------------------------------------------------------
def test_germany50_routes_match_networkx_on_sampled_pairs(germany):
    G = to_networkx(germany)
    rng = random.Random(42)
    for a, b in rng.sample(list(itertools.permutations(germany.nodes, 2)), 40):
        ref = [nx.path_weight(G, p, "weight")
               for p in itertools.islice(nx.shortest_simple_paths(G, a, b, weight="weight"), 4)]
        got = k_shortest_paths(germany, a, b, 4)
        check_routes(germany, got, a, b)
        assert [l for _, l in got] == pytest.approx(ref), (a, b)
