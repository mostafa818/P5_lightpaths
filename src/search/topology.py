"""Topology parser and link lengths (plan Steps 1.6b and 1.7b, Member B).

Reads an SNDlib native-format network file (Abilene), builds the undirected
graph and computes each link's length in km with the haversine formula.

Usage:
    from search.topology import load_topology
    topo = load_topology("data/raw/topology/abilene_sndlib_native.txt")
    topo.nodes                 # {"ATLAM5": (lon, lat), ...}
    topo.links                 # {("ATLAM5", "ATLAng"): length_km, ...}
    topo.adjacency()           # {"ATLAM5": [("ATLAng", 123.4), ...], ...}
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from common import link_id

EARTH_RADIUS_KM = 6371.0088

_NUM = r"-?\d+(?:\.\d+)?"
_NODE_RE = re.compile(rf"^(\S+)\s*\(\s*({_NUM})\s+({_NUM})\s*\)")   # NAME ( lon lat )
_LINK_RE = re.compile(r"^(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)")          # ID ( end1 end2 ) ...


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two (longitude, latitude) points in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Topology:
    nodes: dict   # node name -> (longitude, latitude)
    links: dict   # undirected link id (sorted tuple) -> length_km (our haversine value)
    declared_km: dict | None = None   # germany50 only: the file's own `dist` per link, for cross-checks

    def adjacency(self) -> dict:
        """node -> list of (neighbour, length_km), sorted for deterministic order."""
        adj = {n: [] for n in self.nodes}
        for (u, v), length in self.links.items():
            adj[u].append((v, length))
            adj[v].append((u, length))
        for n in adj:
            adj[n].sort()
        return adj

    def distance_km(self, a: str, b: str) -> float:
        """Straight-line (haversine) distance between two nodes; the A* heuristic."""
        (lon1, lat1), (lon2, lat2) = self.nodes[a], self.nodes[b]
        return haversine_km(lon1, lat1, lon2, lat2)


def _section_lines(text: str, name: str) -> list:
    """Content lines of a `NAME ( ... )` section, without blanks and # comments."""
    m = re.search(rf"^\s*{name}\s*\(\s*$(.*?)^\s*\)\s*$", text, re.MULTILINE | re.DOTALL)
    if m is None:
        raise ValueError(f"section {name} not found")
    lines = (ln.strip() for ln in m.group(1).splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def parse_sndlib(text: str) -> Topology:
    """Parse the NODES and LINKS sections of an SNDlib native file.

    Other sections (META, DEMANDS, ADMISSIBLE_PATHS) are ignored.
    Raises ValueError on malformed lines, unknown endpoints, self-loops or
    duplicate links (A-B and B-A count as the same cable).
    """
    nodes = {}
    for ln in _section_lines(text, "NODES"):
        m = _NODE_RE.match(ln)
        if m is None:
            raise ValueError(f"bad NODES line: {ln!r}")
        name, lon, lat = m.group(1), float(m.group(2)), float(m.group(3))
        if name in nodes:
            raise ValueError(f"duplicate node {name}")
        nodes[name] = (lon, lat)

    links = {}
    for ln in _section_lines(text, "LINKS"):
        m = _LINK_RE.match(ln)
        if m is None:
            raise ValueError(f"bad LINKS line: {ln!r}")
        u, v = m.group(2), m.group(3)
        for end in (u, v):
            if end not in nodes:
                raise ValueError(f"link {m.group(1)} uses unknown node {end}")
        if u == v:
            raise ValueError(f"self-loop on {u}")
        lid = link_id(u, v)
        if lid in links:
            raise ValueError(f"duplicate link {lid}")
        (lon1, lat1), (lon2, lat2) = nodes[u], nodes[v]
        links[lid] = haversine_km(lon1, lat1, lon2, lat2)

    return Topology(nodes=nodes, links=links)


def load_topology(path) -> Topology:
    """Read and parse an SNDlib native file from disk."""
    return parse_sndlib(Path(path).read_text(encoding="utf-8", errors="replace"))


def parse_node_link_json(data: dict) -> Topology:
    """Parse a networkx node-link JSON network (the TopoHub copy of germany50, plan Step 2.9).

    Nodes are {"id", "name", "pos": [longitude, latitude]}, edges are {"source", "target", "dist"}
    with source/target given as node ids. Node names are used as node identifiers. Link lengths are
    recomputed with haversine from the node positions, so that every link cost equals the
    straight-line distance between its endpoints (this keeps the A* heuristic consistent); the file's
    own `dist` (km) is kept in `declared_km` for comparison.
    """
    by_id = {}
    nodes = {}
    for n in data["nodes"]:
        name = str(n["name"])
        if name in nodes:
            raise ValueError(f"duplicate node name {name}")
        lon, lat = n["pos"]
        by_id[n["id"]] = name
        nodes[name] = (float(lon), float(lat))
    links, declared = {}, {}
    for e in data["edges"]:
        try:
            u, v = by_id[e["source"]], by_id[e["target"]]
        except KeyError as exc:
            raise ValueError(f"edge uses unknown node id {exc}") from None
        if u == v:
            raise ValueError(f"self-loop on {u}")
        lid = link_id(u, v)
        if lid in links:
            raise ValueError(f"duplicate link {lid}")
        (lon1, lat1), (lon2, lat2) = nodes[u], nodes[v]
        links[lid] = haversine_km(lon1, lat1, lon2, lat2)
        if "dist" in e:
            declared[lid] = float(e["dist"])
    return Topology(nodes=nodes, links=links, declared_km=declared or None)


def load_germany50(path) -> Topology:
    """Read the 50-node germany50 network (data/raw/topology/germany50.json)."""
    import json

    return parse_node_link_json(json.loads(Path(path).read_text(encoding="utf-8")))


def write_links_csv(topo: Topology, path) -> None:
    """Save the derived link lengths (plan Step 1.7b: 'save as length_km')."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("u,v,length_km\n")
        for (u, v), length in sorted(topo.links.items()):
            fh.write(f"{u},{v},{length:.3f}\n")


if __name__ == "__main__":
    # python -m search.topology  ->  data/processed/abilene_links.csv
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    topo = load_topology(cfg["paths"]["abilene_topology"])
    out = Path(cfg["paths"]["processed"]) / "abilene_links.csv"
    write_links_csv(topo, out)
    print(f"{len(topo.nodes)} nodes, {len(topo.links)} links, "
          f"total {sum(topo.links.values()):.0f} km -> {out}")
