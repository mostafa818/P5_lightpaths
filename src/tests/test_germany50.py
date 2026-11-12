"""germany50 loading (plan Step 2.9) and the topology JSON parser."""
import statistics

import pytest

from search.topology import load_germany50, parse_node_link_json


def test_parse_node_link_json_small():
    data = {"nodes": [{"id": 0, "name": "A", "pos": [0.0, 0.0]}, {"id": 1, "name": "B", "pos": [0.0, 1.0]}],
            "edges": [{"source": 0, "target": 1, "dist": 111.0}]}
    topo = parse_node_link_json(data)
    assert topo.nodes == {"A": (0.0, 0.0), "B": (0.0, 1.0)}
    assert topo.links[("A", "B")] == pytest.approx(111.2, abs=0.2)     # our haversine, not the file's dist
    assert topo.declared_km == {("A", "B"): 111.0}


@pytest.mark.parametrize("edges,msg", [
    ([{"source": 0, "target": 5}], "unknown node id"),
    ([{"source": 0, "target": 0}], "self-loop"),
    ([{"source": 0, "target": 1}, {"source": 1, "target": 0}], "duplicate link"),
])
def test_parse_node_link_json_errors(edges, msg):
    data = {"nodes": [{"id": 0, "name": "A", "pos": [0, 0]}, {"id": 1, "name": "B", "pos": [0, 1]}],
            "edges": edges}
    with pytest.raises(ValueError, match=msg):
        parse_node_link_json(data)


def test_duplicate_node_name_raises():
    data = {"nodes": [{"id": 0, "name": "A", "pos": [0, 0]}, {"id": 1, "name": "A", "pos": [0, 1]}], "edges": []}
    with pytest.raises(ValueError, match="duplicate node name"):
        parse_node_link_json(data)


def test_germany50_size_and_connectivity(germany):
    assert len(germany.nodes) == 50 and len(germany.links) == 88
    import networkx as nx
    from tests.conftest import to_networkx
    assert nx.is_connected(to_networkx(germany))


def test_germany50_own_haversine_roughly_matches_declared_dist(germany_topo):
    topo = germany_topo
    ratios = [topo.links[l] / topo.declared_km[l] for l in topo.links]
    assert statistics.median(ratios) == pytest.approx(1.0, abs=0.05)
    assert 0.8 < min(ratios) and max(ratios) < 1.2
