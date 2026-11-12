"""Graph class (plan Step 2.1, Member B).

Undirected weighted graph stored as an adjacency dict ``node -> list of (neighbour, length_km)``.
A link id is ``common.link_id(u, v)`` = the sorted pair of node names, so A-B and B-A are the same
cable (this matters later for wavelength conflicts in the CSP).

Node coordinates are optional. When they are present ``straight_line_km`` is the haversine distance
used as the A* heuristic; link lengths built from coordinates (``from_topology``) equal the
straight-line distance between their endpoints, so the heuristic never overestimates and is
consistent. A graph built with ``from_edges`` and no coordinates has heuristic 0 (A* becomes UCS).
"""
from __future__ import annotations

from common import link_id
from search.topology import Topology, haversine_km


class Graph:
    def __init__(self):
        self._adj: dict = {}          # node -> list of (neighbour, length_km), kept sorted
        self._len: dict = {}          # link id -> length_km
        self._pos: dict = {}          # node -> (longitude, latitude)

    # ---- construction -------------------------------------------------------------------------
    @classmethod
    def from_edges(cls, edges, coords: dict | None = None) -> "Graph":
        """edges: iterable of (u, v, length_km). coords: optional node -> (longitude, latitude)."""
        g = cls()
        for n, pos in (coords or {}).items():
            g.add_node(n, pos)
        for u, v, length in edges:
            g.add_link(u, v, length)
        return g

    @classmethod
    def from_topology(cls, topo: Topology) -> "Graph":
        return cls.from_edges(((u, v, d) for (u, v), d in topo.links.items()), coords=topo.nodes)

    def add_node(self, n, pos=None) -> None:
        self._adj.setdefault(n, [])
        if pos is not None:
            self._pos[n] = (float(pos[0]), float(pos[1]))

    def add_link(self, u, v, length_km: float) -> None:
        if u == v:
            raise ValueError(f"self-loop on {u}")
        if length_km < 0:
            raise ValueError(f"negative length on {u}-{v}")
        lid = link_id(u, v)
        if lid in self._len:
            raise ValueError(f"duplicate link {lid}")
        self.add_node(u)
        self.add_node(v)
        self._len[lid] = float(length_km)
        self._adj[u].append((v, float(length_km)))
        self._adj[v].append((u, float(length_km)))
        self._adj[u].sort()
        self._adj[v].sort()

    # ---- queries ------------------------------------------------------------------------------
    @property
    def nodes(self) -> list:
        return list(self._adj)

    @property
    def links(self) -> dict:
        """link id -> length_km (a copy)."""
        return dict(self._len)

    def __contains__(self, n) -> bool:
        return n in self._adj

    def neighbors(self, n) -> list:
        """List of (neighbour, length_km), sorted by neighbour name (deterministic order)."""
        return self._adj[n]

    def length(self, u, v) -> float:
        return self._len[link_id(u, v)]

    def has_link(self, u, v) -> bool:
        return link_id(u, v) in self._len

    def has_coords(self) -> bool:
        return bool(self._adj) and all(n in self._pos for n in self._adj)

    def straight_line_km(self, a, b) -> float:
        """Haversine distance between two nodes, or 0.0 when coordinates are unknown (h = 0)."""
        if a not in self._pos or b not in self._pos:
            return 0.0
        (lon1, lat1), (lon2, lat2) = self._pos[a], self._pos[b]
        return haversine_km(lon1, lat1, lon2, lat2)

    def path_length(self, nodes) -> float:
        """Total length of a node sequence; raises KeyError if two consecutive nodes are not linked."""
        return sum(self.length(a, b) for a, b in zip(nodes, nodes[1:]))

    def path_links(self, nodes) -> frozenset:
        return frozenset(link_id(a, b) for a, b in zip(nodes, nodes[1:]))
