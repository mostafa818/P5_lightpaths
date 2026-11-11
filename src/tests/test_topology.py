"""Tests for the topology parser and link lengths (plan Steps 1.6b, 1.7b)."""
import math
from pathlib import Path

import pytest
import yaml

from common import link_id
from search.topology import haversine_km, load_topology, parse_sndlib

ABILENE = Path("data/raw/topology/abilene_sndlib_native.txt")

SAMPLE = """
META ( unit = MBITPERSEC )
# comment line
NODES (
  A ( 0.0 0.0 )
  B ( 0.0 1.0 )
  C ( 1.0 0.0 )
)
LINKS (
  A_B ( B A ) 1.0 0.0 0.0 0.0 ( 40000.00 1.00 )
  B_C ( C B ) 1.0 0.0 0.0 0.0 ( 40000.00 1.00 )
)
DEMANDS (
  A_C ( A C ) 1 5.00 UNLIMITED
)
"""


def test_haversine_known_values():
    assert haversine_km(10, 20, 10, 20) == 0.0
    # one degree of latitude is about 111.2 km
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.2, abs=0.2)
    # symmetric
    assert haversine_km(-84, 33, -122, 37) == pytest.approx(haversine_km(-122, 37, -84, 33))


def test_parse_sample():
    topo = parse_sndlib(SAMPLE)
    assert set(topo.nodes) == {"A", "B", "C"}
    assert set(topo.links) == {link_id("A", "B"), link_id("B", "C")}
    assert topo.links[("A", "B")] == pytest.approx(111.2, abs=0.2)
    adj = topo.adjacency()
    assert [n for n, _ in adj["B"]] == ["A", "C"]
    assert adj["A"][0][0] == "B"


def test_link_ids_are_undirected_sorted_tuples():
    topo = parse_sndlib(SAMPLE)
    for u, v in topo.links:
        assert u < v


def test_distance_heuristic_never_exceeds_link_length():
    topo = parse_sndlib(SAMPLE)
    for (u, v), length in topo.links.items():
        assert topo.distance_km(u, v) == pytest.approx(length)


def test_unknown_endpoint_raises():
    bad = SAMPLE.replace("B_C ( C B )", "B_C ( C Z )")
    with pytest.raises(ValueError, match="unknown node"):
        parse_sndlib(bad)


def test_duplicate_link_raises():
    bad = SAMPLE.replace("B_C ( C B )", "B_A ( A B )")
    with pytest.raises(ValueError, match="duplicate link"):
        parse_sndlib(bad)


def test_self_loop_raises():
    bad = SAMPLE.replace("B_C ( C B )", "B_B ( B B )")
    with pytest.raises(ValueError, match="self-loop"):
        parse_sndlib(bad)


def test_missing_section_raises():
    with pytest.raises(ValueError, match="LINKS"):
        parse_sndlib("NODES (\n A ( 0 0 )\n)\n")


@pytest.mark.skipif(not ABILENE.exists(), reason="raw Abilene topology not downloaded")
class TestRealAbilene:
    def setup_method(self):
        self.topo = load_topology(ABILENE)

    def test_counts(self):
        assert len(self.topo.nodes) == 12
        assert len(self.topo.links) == 15

    def test_node_order_matches_config(self):
        cfg = yaml.safe_load(open("config.yaml"))
        assert sorted(self.topo.nodes) == sorted(cfg["node_order"])
        assert list(self.topo.nodes) == cfg["node_order"]

    def test_connected_and_degrees(self):
        adj = self.topo.adjacency()
        seen, stack = set(), ["ATLAM5"]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(m for m, _ in adj[n])
        assert seen == set(self.topo.nodes)
        # ATLAM5 hangs off ATLAng only (a leaf), so it has a single route to everything
        leaves = [n for n, v in adj.items() if len(v) == 1]
        assert leaves == ["ATLAM5"]

    def test_lengths_are_plausible(self):
        lengths = self.topo.links
        assert all(0 < d < 3000 for d in lengths.values())
        # ATLAM5-ATLAng is the short Atlanta metro hop, HSTNng-LOSAng the longest
        assert min(lengths, key=lengths.get) == link_id("ATLAM5", "ATLAng")
        assert max(lengths, key=lengths.get) == link_id("HSTNng", "LOSAng")
        assert lengths[link_id("NYCMng", "WASHng")] == pytest.approx(335, abs=5)
        assert math.isfinite(sum(lengths.values()))
