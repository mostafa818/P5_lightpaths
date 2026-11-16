"""Shared fixtures for the search tests (toy graph, Abilene, germany50)."""
from pathlib import Path

import networkx as nx
import pytest
import yaml

from search.graph import Graph
from search.topology import load_germany50, load_topology

ABILENE_FILE = Path("data/raw/topology/abilene_sndlib_native.txt")
GERMANY_FILE = Path("data/raw/topology/germany50.json")


@pytest.fixture(scope="session")
def toy():
    """5-node hand-drawn graph (+ isolated node F), explicit lengths, no coordinates (h = 0).

        A --4-- B --4-- C        A-C 10 (the direct, fewest-hops link)
        |       |       |        C-D 1, B-D 2, D-E 3, C-E 8
        +---10--+-------+        F has no links

    Shortest paths by hand: A->C = A-B-D-C (7; A-B-C is 8 and the direct link 10); A->D = A-B-D (6);
    A->E = A-B-D-E (9). All 7 simple A->E paths: ABDE 9, ABCDE 12, ACDE 14, ABDCE 15, ABCE 16,
    ACE 18, ACBDE 19.
    """
    g = Graph.from_edges([("A", "B", 4), ("B", "C", 4), ("A", "C", 10), ("C", "D", 1),
                          ("B", "D", 2), ("D", "E", 3), ("C", "E", 8)])
    g.add_node("F")
    return g


@pytest.fixture(scope="session")
def abilene():
    if not ABILENE_FILE.exists():
        pytest.skip("raw Abilene topology not downloaded (copy it from the raw-data branch)")
    return Graph.from_topology(load_topology(ABILENE_FILE))


@pytest.fixture(scope="session")
def germany_topo():
    if not GERMANY_FILE.exists():
        pytest.skip("raw germany50 topology not downloaded (copy it from the raw-data branch)")
    return load_germany50(GERMANY_FILE)


@pytest.fixture(scope="session")
def germany(germany_topo):
    return Graph.from_topology(germany_topo)


@pytest.fixture(scope="session")
def reach_km():
    with open("config.yaml") as fh:
        return yaml.safe_load(fh)["rwa"]["reach_km"]


def to_networkx(graph: Graph) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)
    for (u, v), d in graph.links.items():
        G.add_edge(u, v, weight=d)
    return G
