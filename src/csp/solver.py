"""Backtracking, forward checking and AC-3 (plan Steps 3.2-3.5 and 3.7, Member B).

``solve(requests, routes_by_request, W, use_fc, use_ac3, max_assignments)`` returns a
``common.Result``. The three parts are independent switches:
  * plain backtracking (BT): static variable order (highest predicted demand first), values tried
    shortest route first then lowest wavelength, consistency checked against assigned requests only;
  * ``use_fc``: forward checking inside the search. After assigning (route, wavelength) to a request,
    delete from every later neighbour the values that share a link with that route and use the same
    wavelength; an emptied domain undoes the assignment at once. A trail restores the deletions;
  * ``use_ac3``: AC-3 once before the search. BT+FC+AC-3 means AC-3 as preprocessing and FC inside.
All variants use the same orders, so they differ only in speed, never in the answer they find.

Counters (``common.Stats``, defined in the README):
  * assignments: every time a value is actually assigned to a variable (it passed the consistency check,
    or in forward checking it was still in the domain). Values rejected by the check are not counted.
  * backtracks:  every time a variable has no value left and the search returns to the previous variable.
    A value undone because forward checking emptied a domain is not a backtrack (the variable still has
    values to try).
  * time:        wall-clock seconds in ``solve``.
``max_assignments`` is a safety cap: the search stops with status "timeout" before it would make
assignment number ``max_assignments + 1``.
"""
from __future__ import annotations

import time
from collections import deque

from common import Result, Stats
from csp.model import Model


class _Timeout(Exception):
    pass


def _revise(m: Model, dom: list, i: int, j: int) -> bool:
    """Remove the values of i that have no compatible value in j. Return True if i lost values.

    (r, w) of i is supported by j if j has some value (r2, w2) with w2 != w, or with w2 == w and route r2
    not sharing a link with route r."""
    conf = m.conf[i][j]
    dj = dom[j]
    nonempty = [d != 0 for d in dj]
    total = sum(nonempty)
    changed = False
    for w in range(m.W):
        bits = dom[i][w]
        if not bits:
            continue
        other_support = total - nonempty[w] > 0          # a value of j on a different wavelength
        if other_support:
            continue                                      # every value of i on wavelength w is supported
        keep = bits
        for r in range(m.nr[i]):
            if bits >> r & 1 and not (dj[w] & ~conf[r]):  # same wavelength only: needs a non-conflicting route
                keep &= ~(1 << r)
        if keep != bits:
            dom[i][w] = keep
            changed = True
    return changed


def ac3(m: Model, dom: list) -> bool:
    """Enforce arc consistency in place. Return False as soon as some domain becomes empty."""
    queue = deque((i, j) for i in range(m.n) for j in m.neigh[i])
    queued = set(queue)
    while queue:
        i, j = queue.popleft()
        queued.discard((i, j))
        if _revise(m, dom, i, j):
            if not any(dom[i]):
                return False
            for k in m.neigh[i]:
                if k != j and (k, i) not in queued:
                    queue.append((k, i))
                    queued.add((k, i))
    return True


def _search(m: Model, dom: list, stats: Stats, use_fc: bool, cap: int):
    """Depth-first search. Returns values[v] = (route position, wavelength) or None."""
    n, W = m.n, m.W
    values: list = [None] * n

    def rec(v: int) -> bool:
        if v == n:
            return True
        for r in range(m.nr[v]):                              # shortest route first
            for w in range(W):                                # then lowest wavelength
                if not dom[v][w] >> r & 1:
                    continue
                if not use_fc:
                    clash = False
                    for u in m.back[v]:                       # consistency against assigned requests only
                        ru, wu = values[u]
                        if wu == w and m.conf[v][u][r] >> ru & 1:
                            clash = True
                            break
                    if clash:
                        continue
                if stats.assignments >= cap:
                    raise _Timeout
                stats.assignments += 1
                values[v] = (r, w)
                trail, ok = [], True
                if use_fc:
                    for u in m.fwd[v]:
                        old = dom[u][w]
                        new = old & ~m.conf[v][u][r]
                        if new != old:
                            trail.append((u, old))
                            dom[u][w] = new
                            if not any(dom[u]):               # domain wipe-out: undo at once
                                ok = False
                                break
                if ok and rec(v + 1):
                    return True
                for u, old in reversed(trail):
                    dom[u][w] = old
                values[v] = None
        stats.backtracks += 1                                 # no value left for this variable
        return False

    return values if rec(0) else None


def solve(requests: list, routes_by_request: dict, W: int, use_fc: bool = False,
          use_ac3: bool = False, max_assignments: int = 1_000_000) -> Result:
    """Assign a (route, wavelength) to every request, or report why that is impossible.

    ``routes_by_request`` maps request id -> list of ``Route``; the returned assignment maps request id ->
    (index into that list, wavelength). ``status`` is "ok", "no_solution" or "timeout".
    """
    stats = Stats()
    t0 = time.perf_counter()

    def finish(assignment, reason="", status="ok") -> Result:
        stats.time = time.perf_counter() - t0
        return Result(assignment=assignment, stats=stats, reason=reason, status=status)

    for r in requests:                                        # a request without any route can never be served
        if not routes_by_request.get(r.id):
            return finish(None, f"No solution: no route within reach for request {r.id}", "no_solution")
    m = Model(requests, routes_by_request, W)
    if m.n == 0:
        return finish({})
    dom = m.initial_domains()
    if use_ac3 and not ac3(m, dom):
        return finish(None, m.explain_failure(), "no_solution")
    try:
        values = _search(m, dom, stats, use_fc, max_assignments)
    except _Timeout:
        return finish(None, f"Timeout: reached max_assignments={max_assignments}", "timeout")
    if values is None:
        return finish(None, m.explain_failure(), "no_solution")
    return finish(m.to_assignment(values))
