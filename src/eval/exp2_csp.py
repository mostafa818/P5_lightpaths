"""Experiment 2 (plan 7.2, Member B): BT vs BT+FC vs BT+FC+AC-3.

Instances: K in {10, 20, 40} x W in {4, 8} x 10 test hours (504 + 17 i, i = 0..9, spread over the test week, a
different hour of day and weekday each). Requests = top-K pairs by *predicted* demand (NN forecast, seed 0), routes =
3 shortest loop-free routes within 5,000 km, priority = predicted demand. The same instance goes to all three
methods, called directly with ``precheck=False`` (the precheck is a separate optional speed-up, not one of the
compared parts). A second run uses the true demand of the hour instead of the forecast (sanity check).

Per instance we record status ("ok", "no_solution", "timeout"), assignments, backtracks, time and the number of
requests the planner would block (Stage 5 feedback loop, fc+ac3, precheck on). A timeout stops a run at
``max_assignments`` (config: 1,000,000), so its counts are lower bounds; they are reported as timeouts, not hidden
inside the means. Every returned solution is re-checked with ``is_valid``; methods that finish must agree.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import Stats
from csp.model import Model, is_valid
from csp.solver import ac3, solve

from eval.common_eval import CFG, COLORS, md_table, new_fig, out_dir, pm, style
from integration.pipeline import default_planner, nn_forecast_fn, plan, select_requests

METHODS = {"BT": (False, False), "BT+FC": (True, False), "BT+FC+AC3": (True, True)}
KS, WS = [10, 20, 40], [4, 8]
HOURS = [504 + 17 * i for i in range(10)]
CAP = CFG["csp"]["max_assignments"]


def ac3_pruned_share(m: Model) -> float:
    """Share of (route, wavelength) values AC-3 removes before any search (or 1.0 if it empties a domain)."""
    dom = m.initial_domains()
    total = sum(m.nr[v] * m.W for v in range(m.n))
    if not ac3(m, dom):
        return 1.0
    left = sum(bin(b).count("1") for d in dom for b in d)
    return 1 - left / total


def instance(planner, t: int, K: int, pred: np.ndarray) -> tuple:
    reqs = select_requests(pred, planner.pairs, K)
    return reqs, {r.id: planner.routes(r, CFG["rwa"]["routes_per_request"]) for r in reqs}


def run(source: str = "nn", ks=None, ws=None) -> pd.DataFrame:
    planner, H, cfg = default_planner()
    fc = nn_forecast_fn(0)
    rows = []
    for t in HOURS:
        pred = fc(t) if source == "nn" else H[t]
        for K in (ks or KS):
            reqs, routes = instance(planner, t, K, pred)
            for W in (ws or WS):
                m = Model(reqs, routes, W)
                pr = ac3_pruned_share(m)
                blocked = len(plan(planner, t, pred, K, W, method="fc+ac3", max_assignments=CAP).blocked)
                statuses = {}
                for name, (use_fc, use_ac3) in METHODS.items():
                    res = solve(reqs, routes, W, use_fc=use_fc, use_ac3=use_ac3, max_assignments=CAP, precheck=False)
                    if res.status == "ok":
                        assert is_valid(reqs, res.assignment, W, routes, CFG["rwa"]["reach_km"], require_all=True)
                    statuses[name] = res.status
                    rows.append(dict(source=source, hour=t, K=K, W=W, method=name, status=res.status,
                                     assignments=res.stats.assignments, backtracks=res.stats.backtracks,
                                     time_s=res.stats.time, blocked=blocked, ac3_pruned=pr))
                done = {s for s in statuses.values() if s != "timeout"}
                assert len(done) <= 1, f"methods disagree on {t} K={K} W={W}: {statuses}"
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> str:
    rows = []
    for (K, W), g in df.groupby(["K", "W"]):
        for name in METHODS:
            d = g[g.method == name]
            n = len(d)
            ok, no, to = (d.status == "ok").sum(), (d.status == "no_solution").sum(), (d.status == "timeout").sum()
            lb = "≥" if to else ""
            rows.append([K, W, name, f"{ok}/{no}/{to}",
                         f"{lb}{pm(d.assignments.mean(), d.assignments.std(ddof=1), 0)}",
                         f"{lb}{pm(d.backtracks.mean(), d.backtracks.std(ddof=1), 0)}",
                         f"{lb}{pm(1000 * d.time_s.mean(), 1000 * d.time_s.std(ddof=1), 0)}",
                         pm(d.blocked.mean(), d.blocked.std(ddof=1), 1) if name == "BT" else "",
                         f"{100 * d.ac3_pruned.mean():.1f}%" if name == "BT+FC+AC3" else ""])
    return md_table(["K", "W", "Method", "solved / unsolvable / timeout (of 10)", "Assignments", "Backtracks",
                     "Time (ms)", "Requests blocked by the planner", "AC-3 values pruned"], rows)


def matched_table(df: pd.DataFrame) -> str:
    """Only instances every method finished (no timeout): a fair like-for-like comparison."""
    p = df.pivot_table(index=["source", "hour", "K", "W"], columns="method", values="assignments")
    st = df.pivot_table(index=["source", "hour", "K", "W"], columns="method", values="status", aggfunc="first")
    fin = (st != "timeout").all(axis=1)
    rows = []
    for (K, W), g in p[fin].groupby(level=["K", "W"]):
        rows.append([K, W, len(g)] + [f"{g[m].mean():.0f}" for m in METHODS] +
                    [f"{(g['BT'] / g['BT+FC'].clip(lower=1)).median():.1f}x"])
    return md_table(["K", "W", "Instances all three finished", "BT", "BT+FC", "BT+FC+AC3", "median BT / FC"], rows)


def plot(df: pd.DataFrame, path) -> None:
    fig, axes = new_fig(1, 2, figsize=(12, 4))
    groups = [(K, W) for K in KS for W in WS]
    for ax, (col, ylabel, scale) in zip(axes, [("assignments", "Assignments (mean over 10 hours)", 1),
                                              ("time_s", "Time, ms (mean over 10 hours)", 1000)]):
        for mi, name in enumerate(METHODS):
            xs, ys, tos = [], [], []
            for gi, (K, W) in enumerate(groups):
                d = df[(df.K == K) & (df.W == W) & (df.method == name)]
                xs.append(gi + (mi - 1) * 0.27)
                ys.append(d[col].mean() * scale)
                tos.append((d.status == "timeout").sum())
            ax.bar(xs, ys, width=0.25, color=COLORS[name], label=name)
            for x, y, to in zip(xs, ys, tos):
                if to:
                    ax.text(x, y * 1.15, f"T{to}", ha="center", fontsize=7, color="#0b0b0b")
        ax.set_xticks(range(len(groups)), [f"K={K}\nW={W}" for K, W in groups])
        ax.set_yscale("symlog", linthresh=1)
        style(ax, ylabel=ylabel)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Experiment 2: BT vs BT+FC vs BT+FC+AC-3, NN-forecast instances (log axis; "
                 "T<n> = n of 10 runs hit the 1,000,000-assignment cap, so the bar is a lower bound)",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


SMALL_W = ([10, 20], [1, 2])      # supplementary: where AC-3 can matter (plan step 3.4 note)


def main(reuse: bool = False) -> None:
    """``reuse=True`` (python -m eval.exp2_csp --reuse) rebuilds tables and plot from the saved raw CSV."""
    t0 = time.time()
    od = out_dir()
    if reuse:
        allf = pd.read_csv(od / "exp2_csp_raw.csv")
        write_outputs(allf, od)
        return
    frames = [run("nn"), run("true"), run("nn", *SMALL_W).assign(source="nn_smallW")]
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(od / "exp2_csp_raw.csv", index=False)
    write_outputs(allf, od)
    print(f"[exp2 done in {time.time() - t0:.0f}s]")


def write_outputs(allf: pd.DataFrame, od) -> None:
    parts = []
    for source, label in (("nn", "NN-forecast instances (main result)"), ("true", "true-demand instances (check)"),
                          ("nn_smallW", "supplementary: small W on NN-forecast instances (where AC-3 can prune)")):
        parts.append(f"### {label}\n\n" + summarise(allf[allf.source == source]))
    main_df = allf[allf.source.isin(["nn", "true"])]
    text = ("## Experiment 2: BT vs BT+FC vs BT+FC+AC-3\n\n"
            "Same 10 instances (hours 504 + 17 i) for all three methods, `precheck` off, cap 1,000,000 assignments. "
            "`≥` marks a row with timeouts: those runs stopped at the cap, so the counts are lower bounds. "
            "'Requests blocked by the planner' comes from the Stage 5 feedback loop (BT+FC+AC-3, precheck on) on the "
            "same instance; a solver timeout inside that loop also drops a request, so it can overstate blocking at "
            "K=40, W=8.\n\n" + "\n".join(parts) +
            "\n### Like-for-like (only instances where all three methods finished; NN and true demand together)\n\n"
            + matched_table(main_df))
    (od / "exp2_csp_summary.md").write_text(text)
    plot(allf[allf.source == "nn"], od / "exp2_csp.png")
    print(text)


if __name__ == "__main__":
    import sys
    main(reuse="--reuse" in sys.argv)
