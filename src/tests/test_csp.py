"""CSP: BT, forward checking, AC-3, validator, impossible cases, safety cap (plan Steps 3.2-3.9)."""
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pytest

import common
from common import Request, Route, Result
from csp.model import Model, is_valid, validate
from csp.solver import ac3, solve

VARIANTS = {"BT": (False, False), "BT+AC3": (False, True), "BT+FC": (True, False), "BT+FC+AC3": (True, True)}


def route(length, *links):
    """A Route over named links such as "AB" (undirected link ids are sorted tuples)."""
    ids = frozenset(tuple(sorted(l)) for l in links)
    return Route(nodes=(), links=ids, length_km=float(length))


def req(i, demand=1.0):
    return Request(id=i, src="s", dst="d", demand=demand)


def run(requests, routes, W, variant="BT", **kw):
    fc, a3 = VARIANTS[variant]
    return solve(requests, routes, W, use_fc=fc, use_ac3=a3, **kw)


# ---- hand-made instances ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
def test_requests_sharing_a_link_get_different_wavelengths(variant):
    rs = [req(1, 5), req(2, 3)]
    routes = {1: [route(10, "AB")], 2: [route(10, "AB", "BC")]}
    res = run(rs, routes, 2, variant)
    assert res.status == "ok" and res.assignment == {1: (0, 0), 2: (0, 1)}      # lowest wavelength first
    assert is_valid(rs, res.assignment, 2, routes, require_all=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_link_disjoint_requests_may_share_a_wavelength(variant):
    rs = [req(1), req(2), req(3)]
    routes = {1: [route(1, "AB")], 2: [route(1, "CD")], 3: [route(1, "EF")]}
    res = run(rs, routes, 1, variant)                                           # W = 1
    assert res.assignment == {1: (0, 0), 2: (0, 0), 3: (0, 0)}


def test_shared_link_is_undirected():
    rs = [req(1, 2), req(2, 1)]
    routes = {1: [route(1, "AB")], 2: [route(1, "BA")]}                         # same cable, other direction
    assert run(rs, routes, 1).status == "no_solution"


@pytest.mark.parametrize("variant", VARIANTS)
def test_value_order_prefers_the_shortest_route_even_if_listed_last(variant):
    rs = [req(1)]
    routes = {1: [route(30, "AB"), route(10, "CD"), route(20, "EF")]}
    assert run(rs, routes, 3, variant).assignment == {1: (1, 0)}                # original index 1 is the shortest


def test_variable_order_is_highest_demand_first_ties_keep_input_order():
    m = Model([req("a", 1), req("b", 9), req("c", 5), req("d", 5)], {k: [route(1, "AB")] for k in "abcd"}, 4)
    assert [r.id for r in m.requests] == ["b", "c", "d", "a"]


def test_tied_demands_are_solved_deterministically():
    rs = [req(1, 4.0), req(2, 4.0)]
    routes = {1: [route(1, "AB")], 2: [route(1, "AB")]}
    assert run(rs, routes, 2).assignment == {1: (0, 0), 2: (0, 1)}


def test_single_request_and_empty_input():
    assert run([req(1)], {1: [route(1, "AB")]}, 1).assignment == {1: (0, 0)}
    res = solve([], {}, 4)
    assert res.status == "ok" and res.assignment == {}


# ---- counters, worked out by hand ---------------------------------------------------------------------------------
@pytest.fixture
def needs_backtracking():
    """Request 1 (higher demand) prefers the short route over link X; request 2 only has a route over X; W = 1.
    So 1's first choice leaves nothing for 2, and 1 must switch to its second route."""
    rs = [req(1, 9), req(2, 1)]
    routes = {1: [route(10, "XX"), route(20, "YY")], 2: [route(5, "XX")]}
    return rs, routes


def test_plain_backtracking_counters(needs_backtracking):
    rs, routes = needs_backtracking
    res = run(rs, routes, 1, "BT")
    # assign 1=(r0,w0) [1]; 2 has no consistent value -> backtrack [1]; assign 1=(r1,w0) [2]; assign 2 [3]
    assert res.assignment == {1: (1, 0), 2: (0, 0)}
    assert (res.stats.assignments, res.stats.backtracks) == (3, 1)


def test_forward_checking_counters(needs_backtracking):
    rs, routes = needs_backtracking
    res = run(rs, routes, 1, "BT+FC")
    # assign 1=(r0,w0) [1]; FC empties 2's domain, undone at once (not a backtrack); assign 1=(r1,w0) [2]; assign 2 [3]
    assert res.assignment == {1: (1, 0), 2: (0, 0)}
    assert (res.stats.assignments, res.stats.backtracks) == (3, 0)


def test_ac3_prunes_before_the_search(needs_backtracking):
    rs, routes = needs_backtracking
    res = run(rs, routes, 1, "BT+AC3")
    # AC-3 removes (r0,w0) of request 1 (no compatible value left for 2): assign 1=(r1,w0) [1]; assign 2 [2]
    assert res.assignment == {1: (1, 0), 2: (0, 0)}
    assert (res.stats.assignments, res.stats.backtracks) == (2, 0)
    m = Model(rs, routes, 1)
    dom = m.initial_domains()
    assert ac3(m, dom) and dom[0] == [0b10] and dom[1] == [0b1]


def test_ac3_alone_can_prove_a_problem_unsolvable():
    rs = [req(1, 2), req(2, 1)]
    routes = {1: [route(1, "AB")], 2: [route(1, "AB")]}
    res = run(rs, routes, 1, "BT+AC3")
    assert res.status == "no_solution" and res.stats.assignments == 0           # found before any search


def test_ac3_reaches_a_fixed_point_and_is_never_unsound():
    """After AC-3 no arc can prune anything more (this fails if follow-up arcs are not re-queued), and every
    value AC-3 removed is unused by all solutions (checked against brute force on small instances)."""
    from csp.solver import _revise
    rng = random.Random(3)
    chained = 0
    for _ in range(300):
        n, W = rng.randint(2, 5), rng.randint(1, 2)
        rs, routes = random_instance(rng, n, 3)
        m = Model(rs, routes, W)
        full = m.initial_domains()
        dom = [list(d) for d in full]
        if not ac3(m, dom):
            assert not brute_force_solvable(rs, routes, W)
            continue
        for i in range(m.n):
            for j in m.neigh[i]:
                assert not _revise(m, [list(d) for d in dom], i, j), (rs, routes, W)
        chained += dom != full
        for v in range(m.n):                                   # removed values appear in no solution
            for r in range(m.nr[v]):
                for w in range(W):
                    if dom[v][w] >> r & 1:
                        continue
                    target = m.requests[v].id                  # pin this request to the removed value
                    orig_r = m.orig[v][r]
                    others = [q for q in rs if q.id != target]
                    # no assignment of the others may complete it into a valid solution
                    choices = [[(rr, ww) for rr in range(len(routes[q.id])) for ww in range(W)] for q in others]
                    for combo in itertools.product(*choices):
                        sol = {q.id: c for q, c in zip(others, combo)}
                        sol[target] = (orig_r, w)
                        assert not is_valid(rs, sol, W, routes, require_all=True)
    assert chained > 20                                        # AC-3 really pruned in many of the instances


def test_ac3_prunes_little_with_many_wavelengths():
    rs = [req(i, 10 - i) for i in range(4)]
    routes = {i: [route(1, "AB"), route(2, "CD")] for i in range(4)}
    m = Model(rs, routes, 8)
    dom = m.initial_domains()
    before = [list(d) for d in dom]
    assert ac3(m, dom) and dom == before                                         # W = 8: nothing to remove


def test_stats_time_is_recorded():
    assert run([req(1)], {1: [route(1, "AB")]}, 1).stats.time >= 0


# ---- impossible cases (Step 3.9) -----------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
def test_wavelengths_exhausted_on_a_link(variant):
    rs = [req(1, 3), req(2, 2), req(3, 1)]
    routes = {i: [route(10, "XY", "YZ")] for i in (1, 2, 3)}                    # one route each, all through X-Y and Y-Z
    res = run(rs, routes, 2, variant)                                            # W = 2 < 3 requests
    assert res.assignment is None and res.status == "no_solution"
    assert res.reason == "No solution: wavelengths exhausted on link X-Y"


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_route_within_reach_for_a_request(variant):
    rs = [req(1, 3), req(7, 1)]
    res = run(rs, {1: [route(10, "AB")], 7: []}, 4, variant)
    assert res.assignment is None and res.status == "no_solution"
    assert res.reason == "No solution: no route within reach for request 7"
    assert run(rs, {1: [route(10, "AB")]}, 4, variant).reason == "No solution: no route within reach for request 7"


def test_unsolvable_without_a_single_forced_link_gets_a_generic_message():
    # Three requests, each with two routes; every pair of requests can only avoid a clash on one wavelength,
    # so W = 1 fails, but no single link is forced.
    rs = [req(1, 3), req(2, 2), req(3, 1)]
    routes = {1: [route(1, "AB"), route(2, "CD")], 2: [route(1, "AB"), route(2, "CD")],
              3: [route(1, "AB"), route(2, "CD")]}
    res = run(rs, routes, 1)
    assert res.status == "no_solution"
    assert res.reason == "No solution: wavelengths exhausted (no conflict-free assignment with W=1)"


# ---- safety cap (Step 3.7) ----------------------------------------------------------------------------------------
def test_max_assignments_returns_timeout_with_counters():
    rs = [req(i, 100 - i) for i in range(10)]
    routes = {i: [route(1, "AB" if i % 2 else "CD")] for i in range(10)}
    res = run(rs, routes, 10, max_assignments=5)
    assert res.status == "timeout" and res.assignment is None
    assert res.stats.assignments == 5 and "max_assignments=5" in res.reason


def test_cap_equal_to_the_work_needed_still_solves():
    rs = [req(i, 100 - i) for i in range(10)]
    routes = {i: [route(1, "AB")] for i in range(10)}                            # needs exactly 10 assignments
    assert run(rs, routes, 10, max_assignments=10).status == "ok"
    assert run(rs, routes, 10, max_assignments=9).status == "timeout"


# ---- the validator (Step 3.6) ---------------------------------------------------------------------------------------
@pytest.fixture
def two():
    rs = [req(1), req(2)]
    routes = {1: [route(10, "AB"), route(99, "CD")], 2: [route(10, "AB")]}
    return rs, routes


def test_validator_accepts_good_and_flags_bad_solutions(two):
    rs, routes = two
    assert is_valid(rs, {1: (0, 0), 2: (0, 1)}, 2, routes)
    assert is_valid(rs, {1: (1, 0), 2: (0, 0)}, 2, routes)                       # link-disjoint: same wavelength is fine
    assert "both use wavelength 0 on link A-B" in validate(rs, {1: (0, 0), 2: (0, 0)}, 2, routes)[0]
    assert not is_valid(rs, {1: (0, 0), 2: (0, 2)}, 2, routes)                   # wavelength out of range
    assert not is_valid(rs, {1: (5, 0)}, 2, routes)                              # route index out of range
    assert not is_valid(rs, {9: (0, 0)}, 2, routes)                              # unknown request
    assert is_valid(rs, {1: (0, 0)}, 2, routes)                                  # partial solutions are allowed ...
    assert not is_valid(rs, {1: (0, 0)}, 2, routes, require_all=True)            # ... unless every request is required


def test_validator_checks_the_reach_limit(two):
    rs, routes = two
    assert is_valid(rs, {1: (1, 0), 2: (0, 1)}, 2, routes, reach_km=99)
    assert "exceeds reach" in validate(rs, {1: (1, 0)}, 2, routes, reach_km=98)[0]
    assert is_valid(rs, {1: (0, 0)}, 2, routes, reach_km=10)                     # exactly equal is allowed


def test_input_errors():
    with pytest.raises(ValueError, match="duplicate request ids"):
        solve([req(1), req(1)], {1: [route(1, "AB")]}, 2)
    with pytest.raises(ValueError, match="W must be"):
        solve([req(1)], {1: [route(1, "AB")]}, 0)


def test_common_solve_delegates(needs_backtracking):
    rs, routes = needs_backtracking
    assert common.solve(rs, routes, 1, True, True).assignment == run(rs, routes, 1, "BT+FC+AC3").assignment


# ---- BT, FC and AC-3 agree on many random instances ---------------------------------------------------------------
LINKS = ["AB", "AC", "AD", "BC", "BD", "CD"]


def random_instance(rng, n, max_routes):
    rs = [req(i, rng.choice([1, 2, 3, 4, 5])) for i in range(n)]                # ties on purpose
    routes = {i: [route(rng.randint(1, 9), *rng.sample(LINKS, rng.randint(1, 3)))
                  for _ in range(rng.randint(1, max_routes))] for i in range(n)}
    return rs, routes


def brute_force_solvable(rs, routes, W):
    choices = [[(r, w) for r in range(len(routes[q.id])) for w in range(W)] for q in rs]
    for combo in itertools.product(*choices):
        sol = {q.id: c for q, c in zip(rs, combo)}
        if is_valid(rs, sol, W, routes, require_all=True):
            return True
    return False


def test_variants_agree_with_brute_force_on_small_instances():
    rng = random.Random(1)
    solvable = unsolvable = 0
    for _ in range(300):
        n, W = rng.randint(1, 4), rng.randint(1, 3)
        rs, routes = random_instance(rng, n, 3)
        expected = brute_force_solvable(rs, routes, W)
        for name in VARIANTS:
            res = run(rs, routes, W, name)
            assert (res.status == "ok") == expected, (name, rs, routes, W)
            if res.status == "ok":
                assert is_valid(rs, res.assignment, W, routes, require_all=True)
            else:
                assert res.assignment is None and res.reason.startswith("No solution")
        solvable += expected
        unsolvable += not expected
    assert solvable > 30 and unsolvable > 30                                      # the sample really covers both


def test_variants_find_the_same_solution_and_prune_in_order():
    """Same variable and value orders: every variant returns the same first solution. Forward checking
    only skips dead subtrees, and AC-3 only removes values that no solution uses, so assignments can only go
    down: BT >= BT+FC >= BT+FC+AC3."""
    rng = random.Random(2)
    ok = fewer = 0
    for _ in range(300):
        n, W = rng.randint(3, 8), rng.randint(1, 4)
        rs, routes = random_instance(rng, n, 3)
        res = {name: run(rs, routes, W, name) for name in VARIANTS}
        assert len({r.status for r in res.values()}) == 1
        assert len({json.dumps(r.assignment, sort_keys=True) for r in res.values()}) == 1
        for r in res.values():
            if r.assignment is not None:
                assert is_valid(rs, r.assignment, W, routes, require_all=True)
        assert res["BT"].stats.assignments >= res["BT+FC"].stats.assignments >= res["BT+FC+AC3"].stats.assignments
        assert res["BT"].stats.assignments >= res["BT+AC3"].stats.assignments
        ok += res["BT"].status == "ok"
        fewer += res["BT+FC"].stats.assignments < res["BT"].stats.assignments
    assert ok > 50 and fewer > 10


# ---- real Abilene instances -----------------------------------------------------------------------------------------
META = Path("data/processed/meta.json")
HOURLY = Path("data/processed/hourly.npz")


def real_instance(abilene, hour, K, reach_km, k_routes=3):
    with open(META) as fh:
        pairs = json.load(fh)["pairs"]
    demand = np.load(HOURLY)["hourly"][hour]
    top = np.argsort(-demand, kind="stable")[:K]
    rs = [Request(id=int(i), src=pairs[i][0], dst=pairs[i][1], demand=float(demand[i])) for i in top]
    routes = {r.id: common.k_routes(abilene, r.src, r.dst, k_routes, reach_km) for r in rs}
    return rs, routes


@pytest.mark.parametrize("K,W", [(10, 8), (10, 4), (20, 8), (20, 4)])
def test_real_abilene_instances_all_variants_agree_and_are_valid(abilene, reach_km, K, W):
    for hour in (504, 540, 600):
        rs, routes = real_instance(abilene, hour, K, reach_km)
        res = {name: run(rs, routes, W, name, max_assignments=200_000) for name in VARIANTS}
        done = {n: r for n, r in res.items() if r.status != "timeout"}
        assert done                                                              # FC variants always finish here
        assert len({r.status for r in done.values()}) == 1
        for r in done.values():
            if r.status == "ok":
                assert is_valid(rs, r.assignment, W, routes, reach_km=reach_km, require_all=True)
            else:
                assert r.reason.startswith("No solution")
        assert len({json.dumps(r.assignment, sort_keys=True) for r in done.values()}) == 1


# ---- optional precheck (requested by Member A) --------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
def test_precheck_failure_reports_zero_work_and_the_same_message(variant):
    rs = [req(1, 3), req(2, 2), req(3, 1)]
    routes = {i: [route(10, "XY", "YZ")] for i in (1, 2, 3)}
    off = run(rs, routes, 2, variant)
    on = run(rs, routes, 2, variant, precheck=True)
    assert (on.status, on.assignment, on.reason) == (off.status, off.assignment, off.reason)
    assert on.reason == "No solution: wavelengths exhausted on link X-Y"
    assert on.stats.assignments == 0 and on.stats.backtracks == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_precheck_never_changes_the_answer(variant):
    rng = random.Random(7)
    seen_fail = seen_ok = 0
    for _ in range(400):
        rs, routes = random_instance(rng, rng.randint(1, 6), 3)
        W = rng.randint(1, 3)
        off = run(rs, routes, W, variant)
        on = run(rs, routes, W, variant, precheck=True)
        assert (on.status, on.assignment, on.reason) == (off.status, off.assignment, off.reason)
        assert on.stats.assignments <= off.stats.assignments
        seen_fail += on.status != "ok"
        seen_ok += on.status == "ok"
    assert seen_fail > 40 and seen_ok > 40


@pytest.mark.parametrize("variant", ["BT", "BT+FC"])
def test_precheck_turns_a_timeout_into_an_instant_proof(variant):
    # Five requests with three disjoint routes each (a big search space) plus three low-priority requests that
    # all must use link X-Y; with W = 2 that is impossible, but the search only meets it at the very end.
    rs = [req(i, 100 - i) for i in range(5)] + [req(10 + i, 1) for i in range(3)]
    routes = {i: [route(1, f"a{i}"), route(2, f"b{i}"), route(3, f"c{i}")] for i in range(5)}
    routes.update({10 + i: [route(1, "XY")] for i in range(3)})
    slow = run(rs, routes, 2, variant, max_assignments=50)
    fast = run(rs, routes, 2, variant, max_assignments=50, precheck=True)
    assert slow.status == "timeout"
    assert fast.status == "no_solution" and fast.stats.assignments == 0


def test_common_solve_passes_precheck_through():
    rs = [req(i) for i in (1, 2, 3)]
    routes = {i: [route(10, "XY")] for i in (1, 2, 3)}
    res = common.solve(rs, routes, 2, precheck=True)
    assert res.stats.assignments == 0 and res.reason == "No solution: wavelengths exhausted on link X-Y"
