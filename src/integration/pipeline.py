"""Stage 5 (Member A): NN -> Search -> CSP planner with the feedback loop (plan 5.1-5.6).

Request.id is the index of the OD pair in meta.json["pairs"] (0..131), so results can be scored
against the true demand vector of the same hour.
"""
from __future__ import annotations
import json, logging, os, time
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import yaml

from common import Request, Route
from csp.model import is_valid
from csp.solver import solve
from search.graph import Graph
from search.routes import k_routes
from search.topology import load_topology

log = logging.getLogger("integration")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = {"bt": (False, False), "fc": (True, False), "fc+ac3": (True, True)}   # (use_fc, use_ac3)
NO_ROUTE = "no route within reach"
EXHAUSTED = "wavelengths exhausted"
TIMEOUT = "solver timeout"


@dataclass
class PlanResult:
    hour: int
    K: int
    W: int
    method: str
    strict: bool
    requests: list                      # the K selected requests, sorted by predicted demand (priority)
    status: str = "ok"                  # "ok" | "no_solution" | "timeout" (the last two: strict mode only)
    reason: str = ""                    # CSP message of the failed solve (strict mode failure)
    assignments: dict = field(default_factory=dict)       # request id -> (route index, wavelength)
    routes_by_request: dict = field(default_factory=dict)  # routes used in the final solve
    blocked: list = field(default_factory=list)            # [(Request, reason)]
    iterations: list = field(default_factory=list)         # one dict per solve call (5.4: log every iteration)
    total_assignments: int = 0
    total_backtracks: int = 0
    solver_time: float = 0.0
    total_time: float = 0.0

    @property
    def lightpaths(self) -> list:
        """Readable assignment table: one dict per served request."""
        by_id = {r.id: r for r in self.requests}
        out = []
        for rid, (ri, w) in sorted(self.assignments.items(), key=lambda kv: -by_id[kv[0]].demand):
            route = self.routes_by_request[rid][ri]
            out.append({"id": rid, "src": by_id[rid].src, "dst": by_id[rid].dst, "demand": by_id[rid].demand,
                        "route": route.nodes, "length_km": route.length_km, "wavelength": w})
        return out

    def log_text(self) -> str:
        lines = [f"hour {self.hour}  K={self.K}  W={self.W}  method={self.method}  strict={self.strict}"]
        for it in self.iterations:
            lines.append(f"  iter {it['iteration']}: k={it['k']} requests={it['n_requests']} -> {it['status']}"
                         f" (assignments {it['assignments']}, backtracks {it['backtracks']}) [{it['note']}]"
                         + (f" {it['reason']}" if it["reason"] else ""))
        for r, why in self.blocked:
            lines.append(f"  blocked: {r.src}->{r.dst} (id {r.id}, predicted {r.demand:.1f} Mbps): {why}")
        lines.append(f"  result: {self.status}, served {len(self.assignments)}/{len(self.requests)}"
                     + (f" - {self.reason}" if self.reason else ""))
        return "\n".join(lines)


class Planner:
    """Holds the graph, the pair list and a route cache (routes depend only on src, dst, k)."""

    def __init__(self, graph: Graph, pairs: list, reach_km: float, routes_first: int = 3, routes_max: int = 5):
        self.graph, self.pairs, self.reach_km = graph, [tuple(p) for p in pairs], reach_km
        self.routes_first, self.routes_max = routes_first, routes_max
        self._cache: dict = {}

    def routes(self, req: Request, k: int) -> list:
        key = (req.src, req.dst, k)
        if key not in self._cache:
            self._cache[key] = k_routes(self.graph, req.src, req.dst, k, self.reach_km)
        return self._cache[key]


def default_planner(config_path: str = None) -> tuple:
    """(Planner, H, cfg) built from config.yaml, data/processed and the Abilene topology."""
    cfg = yaml.safe_load(open(config_path or os.path.join(ROOT, "config.yaml")))
    meta = json.load(open(os.path.join(ROOT, cfg["paths"]["processed"], "meta.json")))
    H = np.load(os.path.join(ROOT, cfg["paths"]["processed"], "hourly.npz"))["hourly"]
    g = Graph.from_topology(load_topology(os.path.join(ROOT, cfg["paths"]["abilene_topology"])))
    r = cfg["rwa"]
    return Planner(g, meta["pairs"], r["reach_km"], r["routes_per_request"], r["max_routes"]), H, cfg


# --- forecast sources (Experiment 3 compares these) -----------------------------------------------
def nn_forecast_fn(seed: int = 0) -> Callable:
    from nn.forecast import forecast
    return lambda t: forecast(t, seed=seed)


def last_hour_forecast_fn(H: np.ndarray) -> Callable:
    return lambda t: H[t - 1]


# --- step 5.1 -------------------------------------------------------------------------------------
def select_requests(predicted: np.ndarray, pairs: list, K: int) -> list:
    """Top-K predicted pairs as Requests; predicted demand is the priority. Ties: lower pair index first."""
    top = np.argsort(-np.asarray(predicted), kind="stable")[:K]
    return [Request(int(i), pairs[i][0], pairs[i][1], float(predicted[i])) for i in top]


# --- steps 5.2-5.5 --------------------------------------------------------------------------------
def plan(planner: Planner, t: int, predicted: np.ndarray, K: int, W: int, method: str = "fc+ac3",
         strict: bool = False, max_assignments: int = 1_000_000, precheck: bool = True,
         validate: bool = True) -> PlanResult:
    """Forecast -> top-K requests -> routes -> CSP, with the feedback loop of step 5.4.

    Normal mode: no solution -> retry with more routes (k = max_routes) -> drop the lowest-priority
    request ("wavelengths exhausted", or "solver timeout") and solve again, until a solution exists.
    Requests without any route within reach are blocked up front ("no route within reach").
    If the k=max_routes lists equal the k=first lists, the "more routes" solve is skipped (it would only
    repeat the failure); after the first failure the longer lists stay in use for later solves.
    Strict mode: nothing is dropped or pre-blocked; the CSP's message is returned and a solver timeout
    keeps status "timeout" (it is not a proof of infeasibility).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    use_fc, use_ac3 = METHODS[method]
    t0 = time.perf_counter()
    reqs = select_requests(predicted, planner.pairs, K)
    res = PlanResult(hour=t, K=K, W=W, method=method, strict=strict, requests=reqs)

    k = planner.routes_first
    routes = {r.id: planner.routes(r, k) for r in reqs}
    active = list(reqs)
    if not strict:                                             # 5.2: unroutable requests never enter the CSP
        for r in reqs:
            if not routes[r.id]:
                res.blocked.append((r, NO_ROUTE))
        active = [r for r in reqs if routes[r.id]]

    def attempt(note: str):
        sub = {r.id: routes[r.id] for r in active}
        out = solve(active, sub, W, use_fc=use_fc, use_ac3=use_ac3, max_assignments=max_assignments,
                    precheck=precheck)
        res.iterations.append({"iteration": len(res.iterations) + 1, "k": k, "n_requests": len(active),
                               "status": out.status, "reason": out.reason, "assignments": out.stats.assignments,
                               "backtracks": out.stats.backtracks, "time": out.stats.time, "note": note})
        res.total_assignments += out.stats.assignments
        res.total_backtracks += out.stats.backtracks
        res.solver_time += out.stats.time
        log.info("hour %d iter %d: k=%d n=%d -> %s %s", t, len(res.iterations), k, len(active), out.status, out.reason)
        return out

    note = "first attempt"
    out = None
    while True:
        if not active:
            out = None
            break
        out = attempt(note)
        if out.status == "ok":
            break
        if k < planner.routes_max:                             # 5.4: first try more routes
            more = {r.id: planner.routes(r, planner.routes_max) for r in active}
            same = all(more[r.id] == routes[r.id] for r in active)
            k, routes = planner.routes_max, more               # later solves keep the longer route lists
            if not same:                                       # identical lists would only repeat the failure
                out = attempt("more routes")
                if out.status == "ok":
                    break
        if strict:                                             # 5.5: no dropping, report the CSP message
            res.status, res.reason = out.status, out.reason or f"solver {out.status}"   # "no_solution" or "timeout"
            res.total_time = time.perf_counter() - t0
            return res
        victim = max(active, key=lambda r: (-r.demand, r.id))   # lowest predicted demand = lowest priority
        res.blocked.append((victim, EXHAUSTED if out.status == "no_solution" else TIMEOUT))
        active = [r for r in active if r.id != victim.id]
        note = f"after dropping id {victim.id}"

    res.assignments = dict(out.assignment) if out is not None else {}
    res.routes_by_request = {r.id: routes[r.id] for r in active}
    if validate and active:
        assert is_valid(active, res.assignments, W, res.routes_by_request, planner.reach_km, require_all=True), \
            "BUG: CSP returned an invalid assignment"
    res.total_time = time.perf_counter() - t0
    return res


# --- step 5.6 metrics -----------------------------------------------------------------------------
def evaluate(result: PlanResult, true_demand: np.ndarray) -> dict:
    """served share = true demand of true-top-K pairs that were selected AND assigned / true demand of
    all true-top-K pairs; blocking rate = blocked / K (selected requests); top-K hit rate of the selection."""
    K = len(result.requests)
    true = np.asarray(true_demand, dtype=float)
    true_top = set(np.argsort(-true, kind="stable")[:K].tolist())
    selected = {r.id for r in result.requests}
    served_ids = set(result.assignments) if result.status == "ok" else set()
    den = float(true[list(true_top)].sum())
    return {
        "served_share": float(sum(true[i] for i in true_top & served_ids) / den) if den > 0 else 0.0,
        "blocking_rate": (len(result.blocked) / K) if result.status == "ok" else 1.0,
        "topk_hit_rate": len(true_top & selected) / K if K else 0.0,
        "n_served": len(served_ids), "n_blocked": len(result.blocked),
        "assignments": result.total_assignments, "backtracks": result.total_backtracks,
        "solver_time": result.solver_time,
    }
