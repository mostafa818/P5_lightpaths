"""CSP model and validator for route and wavelength assignment (plan Steps 3.1 and 3.6, Member B).

Variables: one per request, in static order of predicted demand (highest first; ties keep the input
order). Values: (route, wavelength) with wavelength in 0..W-1. Two requests clash when their routes share
a link (undirected link id) and they use the same wavelength.

Speed: the clash test runs millions of times, so everything is precomputed once.
  * Each variable's routes are sorted by length (stable), so route position 0 is the shortest route and
    "shortest route first, then lowest wavelength" is simply the natural loop order. ``orig`` maps a
    sorted position back to the route's index in the caller's list (that is what ``solve`` returns).
  * A domain is a list with one integer bitmask per wavelength: bit r of ``dom[v][w]`` is set when
    (route r, wavelength w) is still possible for variable v.
  * ``conf[i][j][r]`` is the bitmask of the routes of variable j that share a link with route r of
    variable i. Two variables are neighbours only if some pair of their routes shares a link; only
    neighbours appear in ``conf``.
"""
from __future__ import annotations

from typing import Optional


class Model:
    def __init__(self, requests, routes_by_request, W: int):
        if W < 1:
            raise ValueError("W must be at least 1")
        ids = [r.id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate request ids")
        order = sorted(range(len(requests)), key=lambda i: -requests[i].demand)   # stable
        self.W = W
        self.requests = [requests[i] for i in order]
        self.n = len(order)
        self.routes, self.orig = [], []
        for req in self.requests:
            rs = list(routes_by_request.get(req.id, []))
            idx = sorted(range(len(rs)), key=lambda i: rs[i].length_km)           # stable
            self.routes.append([rs[i] for i in idx])
            self.orig.append(idx)
        self.nr = [len(r) for r in self.routes]
        self.conf = [dict() for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                ci = [0] * self.nr[i]
                cj = [0] * self.nr[j]
                for a, ra in enumerate(self.routes[i]):
                    for b, rb in enumerate(self.routes[j]):
                        if ra.links & rb.links:
                            ci[a] |= 1 << b
                            cj[b] |= 1 << a
                if any(ci):
                    self.conf[i][j] = ci
                    self.conf[j][i] = cj
        self.neigh = [sorted(c) for c in self.conf]
        self.back = [[u for u in nb if u < v] for v, nb in enumerate(self.neigh)]   # earlier neighbours
        self.fwd = [[u for u in nb if u > v] for v, nb in enumerate(self.neigh)]    # later neighbours

    def initial_domains(self) -> list:
        return [[(1 << self.nr[v]) - 1] * self.W for v in range(self.n)]

    def to_assignment(self, values: list) -> dict:
        """values[v] = (sorted route position, wavelength) -> {request id: (original route index, wavelength)}."""
        return {self.requests[v].id: (self.orig[v][r], w) for v, (r, w) in enumerate(values)}

    def forced_overload(self):
        """If more requests are forced through one link (every one of their routes uses it) than there are
        wavelengths, return the no-solution message naming that link; otherwise None. Such an instance can
        never be solved, whatever the other requests do."""
        forced: dict = {}
        for v in range(self.n):
            if not self.routes[v]:
                continue
            for lid in frozenset.intersection(*(r.links for r in self.routes[v])):
                forced.setdefault(lid, []).append(v)
        over = [(-len(vs), lid) for lid, vs in forced.items() if len(vs) > self.W]
        if over:
            _, lid = min(over)                      # most requests first, then smallest link id
            return f"No solution: wavelengths exhausted on link {lid[0]}-{lid[1]}"
        return None

    def explain_failure(self) -> str:
        """Clear no-solution message: the overloaded link if there is one, otherwise a generic message."""
        return self.forced_overload() or \
            f"No solution: wavelengths exhausted (no conflict-free assignment with W={self.W})"


def validate(requests, solution: dict, W: int, routes_by_request: dict,
             reach_km: Optional[float] = None, require_all: bool = False) -> list:
    """Return a list of problems (empty when the solution is valid).

    ``solution`` maps request id -> (route_index, wavelength), as in ``common.Result.assignment``.
    Checks: known request ids, route index and wavelength in range, the reach limit (when given), and that
    no two assigned requests share a link on the same wavelength. With ``require_all`` every request must
    be assigned (blocked requests are legitimately absent after the integration loop drops them).
    """
    problems = []
    known = {r.id for r in requests}
    chosen = {}                                    # request id -> (Route, wavelength)
    for rid, value in solution.items():
        if rid not in known:
            problems.append(f"unknown request {rid}")
            continue
        ri, w = value
        routes = routes_by_request.get(rid, [])
        if not (isinstance(ri, int) and 0 <= ri < len(routes)):
            problems.append(f"request {rid}: route index {ri} out of range")
            continue
        if not (isinstance(w, int) and 0 <= w < W):
            problems.append(f"request {rid}: wavelength {w} outside 0..{W - 1}")
            continue
        route = routes[ri]
        if reach_km is not None and route.length_km > reach_km + 1e-9:
            problems.append(f"request {rid}: route of {route.length_km:.1f} km exceeds reach {reach_km} km")
        chosen[rid] = (route, w)
    if require_all:
        problems += [f"request {r.id} is not assigned" for r in requests if r.id not in solution]
    items = list(chosen.items())
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            (ia, (ra, wa)), (ib, (rb, wb)) = items[a], items[b]
            if wa == wb and ra.links & rb.links:
                lid = min(ra.links & rb.links)
                problems.append(f"requests {ia} and {ib} both use wavelength {wa} on link {lid[0]}-{lid[1]}")
    return problems


def is_valid(requests, solution: dict, W: int, routes_by_request: dict,
             reach_km: Optional[float] = None, require_all: bool = False) -> bool:
    """True when ``validate`` finds no problem (plan Step 3.6)."""
    return not validate(requests, solution, W, routes_by_request, reach_km, require_all)
