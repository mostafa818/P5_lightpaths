"""Search trace for the demo (plan 6.3, Member B): the order in which UCS and A* expand nodes for one request.

``expansion_trace`` runs the same code as Experiment 1 (``search.algorithms``, through its ``trace=`` and ``Stats``
hooks), so the counters match the README definitions and the experiment tables. ``format_trace`` prints the two
traces side by side so they fit on a slide.
"""
from __future__ import annotations

from common import Stats
from search.algorithms import astar, bfs, ucs

_ALGOS = {"ucs": ucs, "astar": astar, "bfs": bfs}


def expansion_trace(graph, src: str, dst: str, algo: str) -> dict:
    """Run ``algo`` ("ucs", "astar" or "bfs") and return
    ``dict(order=[expanded nodes in order], path=[nodes], cost_km=float, expanded=int, generated=int)``.
    ``path`` is empty and ``cost_km`` is None when dst cannot be reached."""
    if algo not in _ALGOS:
        raise ValueError(f"algo must be one of {sorted(_ALGOS)}, got {algo!r}")
    order, stats = [], Stats()
    res = _ALGOS[algo](graph, src, dst, stats, order)
    path, cost = res if res else ([], None)
    return dict(order=order, path=list(path), cost_km=cost, expanded=stats.expanded,
                generated=stats.generated, algo=algo, src=src, dst=dst)


def format_trace(trace_ucs: dict, trace_astar: dict) -> str:
    """Two columns, one row per expansion step. A star marks a node on the final path; in the UCS column a
    '-' after the A* trace has ended shows how many expansions the heuristic saved."""
    on_path = set(trace_ucs["path"])
    a, b = trace_ucs["order"], trace_astar["order"]
    w = max([len("UCS (h = 0)")] + [len(n) + 2 for n in a]) + 2
    lines = [f"{'step':>4}  {'UCS (h = 0)':<{w}}A* (haversine h)"]
    for i in range(max(len(a), len(b))):
        cell = lambda order: (order[i] + (" *" if order[i] in on_path else "")) if i < len(order) else "-"
        lines.append(f"{i + 1:>4}  {cell(a):<{w}}{cell(b)}")
    for t, name in ((trace_ucs, "UCS"), (trace_astar, "A* ")):
        cost = "no path" if t["cost_km"] is None else f"{t['cost_km']:.0f} km"
        lines.append(f"{name}: expanded {t['expanded']}, generated {t['generated']}, cost {cost}, "
                     f"path {'-'.join(t['path']) or 'none'}")
    saved = trace_ucs["expanded"] - trace_astar["expanded"]
    lines.append(f"(* = on the final path; A* expanded {saved} fewer node{'s' if saved != 1 else ''} "
                 f"for the same cost)")
    return "\n".join(lines)
