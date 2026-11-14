"""Experiment 1 (plan 7.1, Member B): BFS vs UCS vs A* on Abilene (all 132 ordered pairs) and germany50
(100 random ordered pairs, 5 seeds for the sampled pairs; every timing repeated 5 times).

Counters are deterministic (nodes generated / expanded, README definitions); only the time varies between
repeats. For each pair we also record the cost (km) of the path found and its number of hops.

Spread reported in the summary: ``std over seeds`` for germany50 (each seed = a different sample of 100 pairs);
Abilene has one fixed pair set, so its time spread is over the 5 timing repeats and its counters have no spread.
"""
from __future__ import annotations

import itertools
import random
import time

import numpy as np
import pandas as pd

from common import Stats
from search.algorithms import astar, bfs, ucs
from search.graph import Graph
from search.topology import load_germany50, load_topology

from eval.common_eval import CFG, COLORS, ROOT, md_table, new_fig, out_dir, pm, style

ALGOS = {"BFS": bfs, "UCS": ucs, "A*": astar}
REPEATS = 5
SEEDS = [CFG["seed"] + i for i in range(5)]
N_RANDOM = 100


def graphs() -> dict:
    ab = Graph.from_topology(load_topology(ROOT / CFG["paths"]["abilene_topology"]))
    de = Graph.from_topology(load_germany50(ROOT / CFG["paths"]["germany50_topology"]))
    return {"Abilene": ab, "germany50": de}


def sample_pairs(g: Graph, seed: int, n: int = N_RANDOM) -> list:
    pairs = [(a, b) for a in g.nodes for b in g.nodes if a != b]
    return random.Random(seed).sample(sorted(pairs), n)


def one_run(g: Graph, pairs: list) -> list:
    """One pass over the pairs: counters, cost, hops and the mean wall time of REPEATS calls."""
    rows = []
    for src, dst in pairs:
        for name, fn in ALGOS.items():
            times, st, res = [], None, None
            for _ in range(REPEATS):
                st = Stats()
                res = fn(g, src, dst, st)
                times.append(st.time)
            nodes, cost = res
            rows.append(dict(src=src, dst=dst, algo=name, expanded=st.expanded, generated=st.generated,
                             cost_km=cost, hops=len(nodes) - 1, time_ms=1000 * float(np.mean(times)),
                             time_std_ms=1000 * float(np.std(times))))
    return rows


def run() -> dict:
    gs = graphs()
    frames = []
    ab = gs["Abilene"]
    pairs = [(a, b) for a in ab.nodes for b in ab.nodes if a != b]
    assert len(pairs) == 132
    df = pd.DataFrame(one_run(ab, sorted(pairs)))
    df.insert(0, "seed", -1)
    df.insert(0, "graph", "Abilene")
    frames.append(df)
    for s in SEEDS:
        df = pd.DataFrame(one_run(gs["germany50"], sample_pairs(gs["germany50"], s)))
        df.insert(0, "seed", s)
        df.insert(0, "graph", "germany50")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return {"raw": raw, "graphs": gs}


def summarise(raw: pd.DataFrame, gs: dict) -> tuple:
    """(table 1: per-algorithm means, table 2: paired comparisons) as markdown, plus the numbers."""
    rows1, rows2, nums = [], [], {}
    for gname in ("Abilene", "germany50"):
        d = raw[raw.graph == gname]
        for algo in ALGOS:
            a = d[d.algo == algo]
            if gname == "germany50":                       # spread over the 5 seeds (mean of each sample)
                per = a.groupby("seed")[["expanded", "generated", "cost_km", "hops", "time_ms"]].mean()
                cells = [pm(per[c].mean(), per[c].std(ddof=1), dg) for c, dg in
                         (("expanded", 1), ("generated", 1), ("cost_km", 1), ("hops", 2), ("time_ms", 3))]
            else:                                          # counters fixed; time spread over the pairs' repeats
                tstd = float(a.time_std_ms.mean())
                cells = [f"{a.expanded.mean():.1f}", f"{a.generated.mean():.1f}", f"{a.cost_km.mean():.1f}",
                         f"{a.hops.mean():.2f}", pm(a.time_ms.mean(), tstd, 3)]
            rows1.append([gname, algo] + cells)
            nums[(gname, algo)] = a
        # paired comparisons, per (seed, pair)
        key = ["seed", "src", "dst"]
        w = d.pivot_table(index=key, columns="algo", values=["expanded", "generated", "cost_km", "hops"])
        n = len(w)
        ex_u, ex_a = w["expanded"]["UCS"], w["expanded"]["A*"]
        cost_o = w["cost_km"]["UCS"]
        rows2.append([gname, "A* vs UCS", f"{n}",
                      f"A* fewer expanded in {(ex_a < ex_u).mean() * 100:.1f}%, equal {(ex_a == ex_u).mean() * 100:.1f}%, "
                      f"more {(ex_a > ex_u).mean() * 100:.1f}%; mean ratio expanded A*/UCS = {(ex_a / ex_u).mean():.3f}; "
                      f"cost equal in {(np.isclose(w['cost_km']['A*'], cost_o)).mean() * 100:.1f}%"])
        bc, bh = w["cost_km"]["BFS"], w["hops"]["BFS"]
        longer = ~np.isclose(bc, cost_o)
        rows2.append([gname, "BFS vs UCS", f"{n}",
                      f"BFS path longer than optimal in {longer.mean() * 100:.1f}% of pairs "
                      f"(mean excess {((bc / cost_o - 1) * 100).mean():.1f}% overall, "
                      f"{((bc / cost_o - 1) * 100)[longer].mean() if longer.any() else 0:.1f}% when longer); "
                      f"BFS has fewer hops than the UCS path in {(bh < w['hops']['UCS']).mean() * 100:.1f}%, "
                      f"equal {(bh == w['hops']['UCS']).mean() * 100:.1f}%, more {(bh > w['hops']['UCS']).mean() * 100:.1f}%"])
    t1 = md_table(["Graph", "Algorithm", "Expanded", "Generated", "Path cost (km)", "Hops", "Time (ms)"], rows1)
    t2 = md_table(["Graph", "Comparison", "Pairs", "Result"], rows2)
    return t1, t2


def check_bfs_fewest_hops(raw: pd.DataFrame, gs: dict) -> int:
    """Independent check: BFS hop count equals the unweighted shortest-path hop count (networkx)."""
    import networkx as nx
    bad = 0
    for gname, g in gs.items():
        G = nx.Graph()
        for (u, v), _ in g.links.items():
            G.add_edge(u, v)
        d = raw[(raw.graph == gname) & (raw.algo == "BFS")].drop_duplicates(["src", "dst"])
        for r in d.itertuples():
            bad += nx.shortest_path_length(G, r.src, r.dst) != r.hops
    return bad


def plot(raw: pd.DataFrame, path) -> None:
    fig, axes = new_fig(1, 3, figsize=(12, 3.8))
    panels = [("expanded", "Nodes expanded (mean per pair)"), ("generated", "Nodes generated (mean per pair)"),
              ("cost_km", "Path cost, km (mean per pair)")]
    for ax, (col, title) in zip(axes, panels):
        for gi, gname in enumerate(("Abilene", "germany50")):
            d = raw[raw.graph == gname]
            for ai, algo in enumerate(ALGOS):
                a = d[d.algo == algo]
                if gname == "germany50":
                    per = a.groupby("seed")[col].mean()
                    m, s = per.mean(), per.std(ddof=1)
                else:
                    m, s = a[col].mean(), 0.0
                x = gi + (ai - 1) * 0.26
                ax.bar(x, m, width=0.24, color=COLORS[algo], label=algo if gi == 0 else None)
                if s:
                    ax.errorbar(x, m, yerr=s, color="#0b0b0b", capsize=3, linewidth=1)
        ax.set_xticks([0, 1], ["Abilene\n(132 pairs)", "germany50\n(100 pairs x 5 seeds)"])
        style(ax, title=title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 1: BFS vs UCS vs A* (error bars = std over 5 seeds; none for Abilene)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    t0 = time.time()
    res = run()
    raw, gs = res["raw"], res["graphs"]
    od = out_dir()
    raw.to_csv(od / "exp1_search_raw.csv", index=False)
    t1, t2 = summarise(raw, gs)
    bad = check_bfs_fewest_hops(raw, gs)
    text = ("## Experiment 1: BFS vs UCS vs A*\n\n" + t1 + "\n" + t2 +
            f"\nIndependent check: BFS hop count equals the networkx unweighted shortest path in "
            f"{'all' if bad == 0 else 'NOT all'} pairs ({bad} mismatches).\n")
    (od / "exp1_search_summary.md").write_text(text)
    plot(raw, od / "exp1_search.png")
    print(text)
    print(f"[exp1 done in {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
