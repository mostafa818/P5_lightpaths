"""Interface contract shared by A (data/NN/integration) and B (search/CSP).
Agree on any change here before coding against it."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass(frozen=True)
class Request:
    id: int
    src: str
    dst: str
    demand: float          # predicted demand (priority), Mbps


@dataclass(frozen=True)
class Route:
    nodes: tuple           # ordered node names
    links: frozenset       # undirected link ids: tuple(sorted((u, v)))
    length_km: float


@dataclass
class Stats:
    # search counters
    expanded: int = 0      # popped from frontier and processed
    generated: int = 0     # pushed onto frontier
    # CSP counters
    assignments: int = 0   # every tentative variable=value attempt
    backtracks: int = 0    # variable has no value left, search returns
    time: float = 0.0      # seconds


@dataclass
class Result:
    assignment: Optional[dict]   # request id -> (route_index, wavelength), or None
    stats: Stats = field(default_factory=Stats)
    reason: str = ""             # e.g. "No solution: wavelengths exhausted on link X-Y"
    status: str = "ok"           # "ok" | "no_solution" | "timeout"


def link_id(u: str, v: str) -> tuple:
    return tuple(sorted((u, v)))


# --- B implements ---------------------------------------------------------
def k_routes(graph: Any, src: str, dst: str, k: int, reach_km: float) -> list:
    """Up to k loop-free routes (Yen), each with length_km <= reach_km."""
    raise NotImplementedError


def solve(requests: list, routes_by_request: dict, W: int,
          use_fc: bool = False, use_ac3: bool = False,
          max_assignments: int = 1_000_000) -> Result:
    raise NotImplementedError


# --- A implements ---------------------------------------------------------
def forecast(t: int) -> np.ndarray:
    """132 non-negative predicted demands for hour t, using data up to t-1 only.
    Delegates to nn.forecast (extra `seed` argument there; default 0)."""
    from nn.forecast import forecast as _forecast
    return _forecast(t)
