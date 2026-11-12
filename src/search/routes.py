"""Up to k loop-free routes per request with Yen's algorithm (plan Step 2.7, Member B).

Route 1 is the shortest path. Each next route comes from "spur" paths: for every earlier route and
every node on it (the spur node), the links already used from that node (by earlier routes sharing the
same root) and the nodes of the root path are removed, the shortest path from the spur node to the
destination is found, joined to the root, and kept as a candidate. The cheapest unused candidate becomes
the next route. Simple paths are loop-free by construction.

Reach limit: a route is kept only if its total length is <= ``reach_km`` (inclusive; a small float
tolerance is used). Routes come out in non-decreasing length, so the search stops at the first route that
is too long.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Optional

from common import Route, Stats, link_id
from search.algorithms import shortest_path

REACH_TOL = 1e-9


def k_shortest_paths(graph, src, dst, k: int, reach_km: Optional[float] = None,
                     stats: Optional[Stats] = None) -> list:
    """Up to k loop-free paths, shortest first, each as (nodes, length_km), none longer than reach_km.

    Raises ValueError for src == dst, unknown nodes or k < 1. Returns [] when dst is unreachable or
    even the shortest path exceeds the reach limit; returns fewer than k when fewer exist.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    stats = stats if stats is not None else Stats()
    first = shortest_path(graph, src, dst, stats)          # validates src/dst
    if first is None or (reach_km is not None and first[1] > reach_km + REACH_TOL):
        return []
    accepted = [first]
    seen = {tuple(first[0])}
    candidates: list = []                                  # heap of (length, tie, nodes tuple)
    tie = itertools.count()
    while len(accepted) < k:
        prev = accepted[-1][0]
        for i in range(len(prev) - 1):
            spur, root = prev[i], prev[: i + 1]
            banned_links = {link_id(p[i], p[i + 1]) for p, _ in accepted
                            if len(p) > i + 1 and p[: i + 1] == root}
            spur_path = shortest_path(graph, spur, dst, stats, banned_nodes=set(root[:-1]),
                                      banned_links=banned_links)
            if spur_path is None:
                continue
            total = tuple(root[:-1]) + tuple(spur_path[0])
            if total in seen:
                continue
            seen.add(total)
            heapq.heappush(candidates, (graph.path_length(total), next(tie), total))
        if not candidates:
            break
        length, _, nodes = heapq.heappop(candidates)
        if reach_km is not None and length > reach_km + REACH_TOL:
            break                                          # everything left is at least as long
        accepted.append((list(nodes), length))
    return accepted


def k_routes(graph, src, dst, k: int, reach_km: float, stats: Optional[Stats] = None) -> list:
    """Interface contract (common.k_routes): up to k loop-free ``Route`` objects within reach."""
    return [Route(nodes=tuple(nodes), links=graph.path_links(nodes), length_km=length)
            for nodes, length in k_shortest_paths(graph, src, dst, k, reach_km, stats)]
