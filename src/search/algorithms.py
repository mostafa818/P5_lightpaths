"""BFS, UCS and A* (plan Steps 2.2-2.5, Member B).

All three take a ``search.graph.Graph``, a source and a destination, and return ``(nodes, length_km)``
for the path found, or ``None`` when the destination is unreachable. ``src == dst`` and unknown nodes
raise ``ValueError``.

Counters (``common.Stats``, defined in the README; every search counts the same way):
  * generated: every node pushed onto the frontier (the source counts as generated).
  * expanded:  every node popped from the frontier and processed. The goal counts as expanded when it
               is popped (the search stops at pop, not at push). In UCS/A* a stale queue entry for a
               node that is already closed is discarded without counting.
  * time:      wall-clock seconds spent in the call, added to ``stats.time``.

UCS and A* are one function (``shortest_path``): UCS orders the frontier by ``g`` (h = 0), A* by
``f = g + h`` with ``h`` = haversine distance to the destination. Each link cost equals the
straight-line distance between its endpoints, so ``h`` never overestimates and is consistent: A* is
optimal and expands no more nodes than UCS (up to ties).

``banned_nodes`` / ``banned_links`` exclude parts of the graph for one call; Yen's algorithm uses them
for its spur paths. ``trace``, when given, receives the expanded nodes in order (``run_demo --trace``).
"""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from typing import Optional

from common import Stats, link_id


def _check(graph, src, dst) -> None:
    if src == dst:
        raise ValueError(f"source and destination are the same node: {src!r}")
    for n in (src, dst):
        if n not in graph:
            raise ValueError(f"unknown node {n!r}")


def _rebuild(parent: dict, dst) -> list:
    path = [dst]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def bfs(graph, src, dst, stats: Optional[Stats] = None, trace: Optional[list] = None):
    """Breadth-first search: FIFO queue, visited set, parent pointers. Fewest hops, not shortest km.

    Returns (nodes, length_km) where length_km is the sum of the link lengths of the path found.
    """
    _check(graph, src, dst)
    stats = stats if stats is not None else Stats()
    t0 = time.perf_counter()
    parent = {src: None}                     # doubles as the visited set (marked when generated)
    frontier = deque([src])
    stats.generated += 1
    result = None
    while frontier:
        node = frontier.popleft()
        stats.expanded += 1
        if trace is not None:
            trace.append(node)
        if node == dst:
            path = _rebuild(parent, dst)
            result = (path, graph.path_length(path))
            break
        for nbr, _ in graph.neighbors(node):
            if nbr not in parent:
                parent[nbr] = node
                frontier.append(nbr)
                stats.generated += 1
    stats.time += time.perf_counter() - t0
    return result


def shortest_path(graph, src, dst, stats: Optional[Stats] = None, use_heuristic: bool = True,
                  banned_nodes=frozenset(), banned_links=frozenset(), trace: Optional[list] = None):
    """UCS (``use_heuristic=False``) or A* (``use_heuristic=True``). Optimal for non-negative costs."""
    _check(graph, src, dst)
    stats = stats if stats is not None else Stats()
    t0 = time.perf_counter()
    h = (lambda n: graph.straight_line_km(n, dst)) if use_heuristic else (lambda n: 0.0)
    tie = itertools.count()                  # deterministic tie-break: insertion order
    best_g = {src: 0.0}
    parent = {src: None}
    closed = set()
    h0 = h(src)
    frontier = [(h0, h0, next(tie), src)]    # (f, h, tie, node); on equal f prefer the smaller h
    stats.generated += 1
    result = None
    while frontier:
        _, _, _, node = heapq.heappop(frontier)
        if node in closed:                   # stale entry: a cheaper one was already processed
            continue
        closed.add(node)
        stats.expanded += 1
        if trace is not None:
            trace.append(node)
        if node == dst:                      # stop at pop (not at push): this is what guarantees optimality
            path = _rebuild(parent, dst)
            result = (path, best_g[dst])
            break
        g = best_g[node]
        for nbr, cost in graph.neighbors(node):
            if nbr in closed or nbr in banned_nodes or link_id(node, nbr) in banned_links:
                continue
            ng = g + cost
            if ng < best_g.get(nbr, float("inf")):
                best_g[nbr] = ng
                parent[nbr] = node
                hn = h(nbr)
                heapq.heappush(frontier, (ng + hn, hn, next(tie), nbr))
                stats.generated += 1
    stats.time += time.perf_counter() - t0
    return result


def ucs(graph, src, dst, stats=None, trace=None):
    """Uniform-cost search (A* with h = 0)."""
    return shortest_path(graph, src, dst, stats, use_heuristic=False, trace=trace)


def astar(graph, src, dst, stats=None, trace=None):
    """A* with the haversine heuristic."""
    return shortest_path(graph, src, dst, stats, use_heuristic=True, trace=trace)
