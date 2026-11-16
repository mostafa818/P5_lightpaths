"""Member B's half of plan Step 1.9: check the traffic-file router order against the topology.

Needs the raw files (git-ignored, see the `raw-data` branch): data/raw/A.txt, data/raw/X01.gz and the
SNDlib sample matrix. The tests are skipped when they are missing.

Check 1 (routing matrix A): its router names, in order, must be node_order; the demand index must be
i*12+j+1 (source-major); the 15 internal links used by A must be exactly the 15 links of the SNDlib
Abilene topology; most pairs must follow the distance-shortest path in our graph (OSPF uses weights, so
a few differ; measured 114 of 132).
Check 2 (SNDlib sample): X01 slot 21 (2004-03-01 01:45) with rows = senders must rank the 132 flows
like the SNDlib sample matrix (Spearman about 1.0), and clearly better than the transposed matrix.
"""
import collections
import gzip
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import yaml
from scipy.stats import spearmanr

from search.topology import load_topology

RAW = Path("data/raw")
TOPO = RAW / "topology" / "abilene_sndlib_native.txt"
A_TXT = RAW / "A.txt"
X01 = RAW / "X01.gz"
SAMPLE = RAW / "topology" / "demandMatrix-abilene-zhang-5min-20040301-0145.txt"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (TOPO, A_TXT, X01, SAMPLE)),
    reason="raw data not downloaded (git-ignored; copy it from the raw-data branch)",
)

with open("config.yaml") as _fh:
    ORDER = yaml.safe_load(_fh)["node_order"]


@pytest.fixture(scope="module")
def topo():
    return load_topology(TOPO)


@pytest.fixture(scope="module")
def routing():
    """A.txt -> (router names in order, demand index -> (src, dst), demand index -> internal directed links)."""
    names, dmd, links = [], {}, collections.defaultdict(list)
    for ln in Path(A_TXT).read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        link, demand, _li, di, _frac = ln.split()
        x, y = link.split(",")
        s, d = demand.split(",")
        for n in (s, d):
            if n not in names:
                names.append(n)
        dmd[int(di)] = (s, d)
        if x != "*" and y != "*":
            links[int(di)].append((x, y))
    return names, dmd, links


def test_a_router_order_and_demand_index(routing):
    names, dmd, _ = routing
    assert len(names) == 12 and len(ORDER) == 12
    assert names == sorted(names)                      # alphabetical, same as node_order
    for i in range(12):
        for j in range(12):
            assert dmd[i * 12 + j + 1] == (names[i], names[j])


def test_a_links_equal_topology_links(topo, routing):
    names, _, links = routing
    amap = dict(zip(names, ORDER))
    used = {tuple(sorted((amap[x], amap[y]))) for p in links.values() for x, y in p}
    assert used == set(topo.links)


def test_a_paths_follow_distance_shortest_paths(topo, routing):
    names, _, links = routing
    amap = dict(zip(names, ORDER))
    G = nx.Graph()
    for (u, v), d in topo.links.items():
        G.add_edge(u, v, weight=d)
    same = 0
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            nxt = {amap[x]: amap[y] for x, y in links[i * 12 + j + 1]}
            s, d = ORDER[i], ORDER[j]
            seq = [s]
            while seq[-1] != d:
                seq.append(nxt[seq[-1]])
            same += seq == nx.shortest_path(G, s, d, weight="weight")
    assert same >= 110, same                            # measured: 114 of 132


def test_x01_slot21_matches_sndlib_sample():
    with gzip.open(X01, "rt") as fh:
        line = [next(fh) for _ in range(22)][21]
    M = np.array(line.split(), float).reshape(144, 5)[:, 0].reshape(12, 12)   # real block, rows = senders
    sample = {}
    for ln in Path(SAMPLE).read_text().splitlines():
        m = re.match(r"\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\d+\s+([\d.]+)", ln)
        if m:
            sample[(m.group(1), m.group(2))] = float(m.group(3))
    pairs = [(ORDER[i], ORDER[j]) for i in range(12) for j in range(12) if i != j]
    assert len(sample) >= 131                           # zero demands are omitted from the sample file
    sv = np.array([sample.get(p, 0.0) for p in pairs])
    x = np.array([M[i, j] for i in range(12) for j in range(12) if i != j])
    xt = np.array([M[j, i] for i in range(12) for j in range(12) if i != j])
    assert spearmanr(x, sv)[0] > 0.999                  # measured 1.0000
    assert spearmanr(xt, sv)[0] < 0.9                   # transposed is clearly worse (0.80)
    ratio = np.median(sv[x > 0] / x[x > 0])
    assert ratio == pytest.approx(100 * 8 / (300 * 1e6), rel=0.01)   # unit formula of Step 1.5
