"""Domain-size analysis for plot (c) (plan 6.2, Member B).

``domain_sizes`` reports, for every request in priority order (highest predicted demand first, the CSP's variable
order), how many (route, wavelength) values are still possible at three points:

  * ``initial``    - routes x W, before any pruning;
  * ``after_ac3``  - after AC-3 (the arc-consistency preprocessing used by BT+FC+AC-3);
  * ``after_fc``   - after forward checking following the first ``fc_steps`` assignments. Default 1: the
    highest-priority request takes its first value (shortest route, lowest wavelength, as in the search) and the
    values that clash with it are deleted from the later neighbours. With ``fc_steps = n`` the first n requests are
    assigned greedily in that way, without backtracking, stopping early if a domain is emptied. An assigned
    request shows size 1.

When AC-3 empties a domain it stops; ``after_ac3`` then holds the sizes at that moment (the emptied request has 0,
the others are partly pruned, so the figure is honest about how far AC-3 got), ``ac3_wiped_out`` is True,
``wiped_out_request`` names the request, and ``after_fc`` is None. Likewise ``fc_wiped_out`` / ``fc_wiped_request``
if forward checking empties a domain.
"""
from __future__ import annotations

from csp.model import Model
from csp.solver import ac3


def _sizes(dom: list) -> list:
    return [sum(bin(b).count("1") for b in d) for d in dom]


def domain_sizes(requests, routes_by_request, W: int, fc_steps: int = 1) -> dict:
    m = Model(requests, routes_by_request, W)
    ids = [r.id for r in m.requests]
    dom = m.initial_domains()
    out = dict(requests=ids, W=W, fc_steps=fc_steps, initial=_sizes(dom), after_ac3=None, after_fc=None,
               ac3_wiped_out=False, wiped_out_request=None, fc_wiped_out=False, fc_wiped_request=None)
    if m.n == 0:
        out.update(after_ac3=[], after_fc=[])
        return out
    ok = ac3(m, dom)
    out["after_ac3"] = _sizes(dom)
    if not ok:
        out["ac3_wiped_out"] = True
        out["wiped_out_request"] = ids[out["after_ac3"].index(0)]
        return out
    assigned = {}
    for v in range(min(fc_steps, m.n)):
        pick = next(((r, w) for r in range(m.nr[v]) for w in range(W) if dom[v][w] >> r & 1), None)
        if pick is None:
            break
        r, w = pick
        assigned[v] = pick
        for u in m.fwd[v]:
            dom[u][w] &= ~m.conf[v][u][r]
            if not any(dom[u]):
                out["fc_wiped_out"], out["fc_wiped_request"] = True, ids[u]
        if out["fc_wiped_out"]:
            break
    sizes = _sizes(dom)
    for v in assigned:
        sizes[v] = 1
    out["after_fc"] = sizes
    return out
