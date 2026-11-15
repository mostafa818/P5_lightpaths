"""BFS, UCS and A* (plan Step 2.6): hand-computed toy cases, networkx cross-checks, edge cases."""
import itertools
import random

import networkx as nx
import pytest

from common import Stats
from search.algorithms import astar, bfs, shortest_path, ucs
from search.graph import Graph
from tests.conftest import to_networkx

ALGOS = {"bfs": bfs, "ucs": ucs, "astar": astar}


def valid_path(graph, nodes, src, dst):
    assert nodes[0] == src and nodes[-1] == dst
    assert len(set(nodes)) == len(nodes)                    # loop-free
    assert all(graph.has_link(a, b) for a, b in zip(nodes, nodes[1:]))


# ---- hand-computed toy graph (conftest.toy) -------------------------------------------------------
@pytest.mark.parametrize("fn", [ucs, astar])
@pytest.mark.parametrize("src,dst,path,cost", [
    ("A", "C", ["A", "B", "D", "C"], 7),       # beats A-B-C (8) and the direct A-C (10)
    ("A", "D", ["A", "B", "D"], 6),
    ("A", "E", ["A", "B", "D", "E"], 9),
    ("E", "A", ["E", "D", "B", "A"], 9),       # undirected: same cable both ways
])
def test_ucs_and_astar_find_the_hand_computed_shortest_path(toy, fn, src, dst, path, cost):
    nodes, length = fn(toy, src, dst)
    assert nodes == path and length == pytest.approx(cost)


def test_bfs_gives_fewest_hops_not_shortest_distance(toy):
    nodes, length = bfs(toy, "A", "C")
    assert nodes == ["A", "C"] and length == 10             # 1 hop, but longer than UCS's 7
    nodes, length = bfs(toy, "A", "E")
    assert nodes == ["A", "C", "E"] and length == 18        # 2 hops; UCS finds 9 over 3 hops


def test_astar_without_coordinates_is_ucs(toy):
    s1, s2 = Stats(), Stats()
    assert ucs(toy, "A", "E", s1) == astar(toy, "A", "E", s2)
    assert (s1.expanded, s1.generated) == (s2.expanded, s2.generated)


# ---- edge cases -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fn", ALGOS.values(), ids=ALGOS.keys())
def test_src_equals_dst_raises(toy, fn):
    with pytest.raises(ValueError, match="same node"):
        fn(toy, "A", "A")


@pytest.mark.parametrize("fn", ALGOS.values(), ids=ALGOS.keys())
def test_unknown_node_raises(toy, fn):
    with pytest.raises(ValueError, match="unknown node"):
        fn(toy, "A", "Z")
    with pytest.raises(ValueError, match="unknown node"):
        fn(toy, "Z", "A")


@pytest.mark.parametrize("fn", ALGOS.values(), ids=ALGOS.keys())
def test_unreachable_returns_none(toy, fn):
    assert fn(toy, "A", "F") is None                        # F has no links


def test_unreachable_counts_the_work_done(toy):
    st = Stats()
    bfs(toy, "A", "F", st)
    assert (st.generated, st.expanded) == (5, 5)            # each of the 5 connected nodes once
    st = Stats()
    ucs(toy, "A", "F", st)
    # expanded: A, B, D, C, E. Generated (hand count): A, B(4), C(10), C(8), D(6), C(7), E(9) = 7,
    # because UCS pushes a node again whenever it finds a cheaper way to it.
    assert (st.generated, st.expanded) == (7, 5)


def test_banned_nodes_and_links_are_respected(toy):
    nodes, length = shortest_path(toy, "A", "E", banned_nodes={"B"})
    assert nodes == ["A", "C", "D", "E"] and length == 14
    nodes, length = shortest_path(toy, "A", "E", banned_links={("B", "D")})
    assert nodes == ["A", "B", "C", "D", "E"] and length == 12
    assert shortest_path(toy, "A", "E", banned_nodes={"B", "C"}) is None


# ---- counters (README definitions) --------------------------------------------------------------------
@pytest.fixture
def star():
    return Graph.from_edges([("X", "W", 1), ("X", "Y", 1), ("X", "Z", 1)])


@pytest.mark.parametrize("fn", [bfs, ucs, astar])
def test_counters_on_a_star_graph(star, fn):
    st = Stats()
    nodes, length = fn(star, "X", "Y", st)
    assert nodes == ["X", "Y"] and length == 1
    assert st.generated == 4          # X, then its neighbours W, Y, Z
    assert st.expanded == 3           # X, W, then the goal Y when it is popped
    assert st.time >= 0


def test_stats_accumulate_across_calls(star):
    st = Stats()
    bfs(star, "X", "Y", st)
    bfs(star, "X", "Y", st)
    assert (st.generated, st.expanded) == (8, 6)


def test_trace_lists_expansions_in_order(toy):
    trace = []
    ucs(toy, "A", "C", trace=trace)
    assert trace[0] == "A" and trace[-1] == "C" and len(set(trace)) == len(trace)
    assert trace == ["A", "B", "D", "C"]    # g: A 0, B 4, D 6, then C 7 (found via D, beats 8 and 10)


# ---- Abilene: all 132 ordered pairs against networkx ------------------------------------------------------
def test_abilene_ucs_and_astar_match_networkx_on_all_pairs(abilene):
    G = to_networkx(abilene)
    pairs = [(a, b) for a in abilene.nodes for b in abilene.nodes if a != b]
    assert len(pairs) == 132
    ucs_exp = astar_exp = 0
    for a, b in pairs:
        ref = nx.shortest_path_length(G, a, b, weight="weight")
        su, sa = Stats(), Stats()
        pu, lu = ucs(abilene, a, b, su)
        pa, la = astar(abilene, a, b, sa)
        assert lu == pytest.approx(ref) and la == pytest.approx(ref)
        valid_path(abilene, pu, a, b)
        valid_path(abilene, pa, a, b)
        assert abilene.path_length(pa) == pytest.approx(la)
        assert sa.expanded <= su.expanded, (a, b)           # consistent heuristic: never expands more
        ucs_exp += su.expanded
        astar_exp += sa.expanded
    assert astar_exp < ucs_exp


def test_abilene_bfs_hops_match_networkx_on_all_pairs(abilene):
    G = to_networkx(abilene)
    longer = 0
    for a, b in itertools.permutations(abilene.nodes, 2):
        nodes, length = bfs(abilene, a, b)
        valid_path(abilene, nodes, a, b)
        assert len(nodes) - 1 == nx.shortest_path_length(G, a, b)
        assert length >= nx.shortest_path_length(G, a, b, weight="weight") - 1e-9
        longer += length > nx.shortest_path_length(G, a, b, weight="weight") + 1e-6
    assert longer > 0                 # fewest hops is not shortest distance (Experiment 1 shows this)


def test_heuristic_is_consistent(abilene, germany):
    """h(u) <= cost(u, v) + h(v) for every link and every destination: A* is optimal."""
    for g in (abilene, germany):
        for (u, v), cost in g.links.items():
            for dst in g.nodes:
                assert g.straight_line_km(u, dst) <= cost + g.straight_line_km(v, dst) + 1e-9
                assert g.straight_line_km(v, dst) <= cost + g.straight_line_km(u, dst) + 1e-9


# ---- germany50 ------------------------------------------------------------------------------------------------
def test_germany50_ucs_and_astar_match_networkx_on_sampled_pairs(germany):
    G = to_networkx(germany)
    rng = random.Random(42)
    pairs = rng.sample(list(itertools.permutations(germany.nodes, 2)), 200)
    ucs_exp = astar_exp = 0
    for a, b in pairs:
        ref = nx.shortest_path_length(G, a, b, weight="weight")
        su, sa = Stats(), Stats()
        _, lu = ucs(germany, a, b, su)
        pa, la = astar(germany, a, b, sa)
        assert lu == pytest.approx(ref) and la == pytest.approx(ref)
        valid_path(germany, pa, a, b)
        ucs_exp += su.expanded
        astar_exp += sa.expanded
    assert astar_exp < ucs_exp


# ---- A* = UCS cost without any raw data (Stage 8: a normal case that still runs in a fresh clone) -------
def test_random_graphs_ucs_astar_and_bfs_match_networkx_without_raw_data():
    """150 seeded random graphs (half with real coordinates, half with many tied integer lengths and none):
    UCS and A* find the networkx optimal cost, BFS the fewest hops, unreachable pairs give None."""
    import random
    import networkx as nx
    from search.graph import Graph, haversine_km

    rng, checked = random.Random(7), 0
    for trial in range(150):
        geo = trial % 2 == 0
        n = rng.randint(4, 12)
        names = [f"n{i}" for i in range(n)]
        pos = {v: (rng.uniform(-120, -70), rng.uniform(25, 48)) for v in names}
        pairs = {tuple(sorted((names[i], names[rng.randrange(i)]))) for i in range(1, n) if rng.random() < 0.9}
        pairs |= {tuple(sorted(rng.sample(names, 2))) for _ in range(rng.randint(0, n))}
        edges = [(a, b, haversine_km(*pos[a], *pos[b]) if geo else float(rng.choice([1, 2, 2, 3, 5]))) for a, b in pairs]
        g = Graph.from_edges(edges, coords=pos if geo else None)
        for v in names:
            g.add_node(v)
        G = nx.Graph()
        G.add_nodes_from(names)
        G.add_weighted_edges_from(edges)
        for _ in range(4):
            s, d = rng.sample(names, 2)
            if not nx.has_path(G, s, d):
                assert ucs(g, s, d) is None and astar(g, s, d) is None and bfs(g, s, d) is None
                continue
            want = nx.shortest_path_length(G, s, d, weight="weight")
            assert ucs(g, s, d)[1] == pytest.approx(want) and astar(g, s, d)[1] == pytest.approx(want)
            assert len(bfs(g, s, d)[0]) - 1 == nx.shortest_path_length(G, s, d)
            checked += 1
    assert checked > 300
