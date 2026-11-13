"""Stage 6.2 plots (Member A): (a) topology + routes coloured by wavelength, (b) link x wavelength heatmap,
(d) NN on vs off. Plot (c), CSP domain sizes, is Member B's (src/eval/plot_csp.py)."""
from __future__ import annotations
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = lambda n: n.replace("ng", "").replace("M5", "-M5")      # ATLAng -> ATLA, ATLAM5 -> ATLA-M5


def _colors(W: int):
    cmap = plt.get_cmap("tab10" if W <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(W)]


def _xy(nodes: dict, n: str, lat0: float = 38.0):
    lon, lat = nodes[n]
    return lon * math.cos(math.radians(lat0)), lat


def plot_topology(result, nodes: dict, links: dict, path: str, title: str = "") -> str:
    """Topology at real coordinates; every served route drawn in the colour of its wavelength. Routes that
    share a link have different wavelengths, so each wavelength gets its own small lateral offset."""
    W = result.W
    cols = _colors(W)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for (u, v) in links:
        (x1, y1), (x2, y2) = _xy(nodes, u), _xy(nodes, v)
        ax.plot([x1, x2], [y1, y2], color="#cccccc", lw=5, solid_capstyle="round", zorder=1)
    delta = 0.20
    for lp in result.lightpaths:
        off = (lp["wavelength"] - (W - 1) / 2) * delta / max(W - 1, 1) * 2
        for a, b in zip(lp["route"], lp["route"][1:]):
            (x1, y1), (x2, y2) = _xy(nodes, a), _xy(nodes, b)
            dx, dy = x2 - x1, y2 - y1
            norm = math.hypot(dx, dy) or 1.0
            px, py = -dy / norm * off, dx / norm * off
            ax.plot([x1 + px, x2 + px], [y1 + py, y2 + py], color=cols[lp["wavelength"]], lw=1.6, zorder=2,
                    alpha=0.95)
    for n in nodes:
        x, y = _xy(nodes, n)
        ax.scatter([x], [y], s=70, color="white", edgecolor="black", zorder=3)
        ax.annotate(SHORT(n), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, zorder=4)
    handles = [plt.Line2D([0], [0], color=cols[w], lw=3, label=f"wavelength {w}") for w in range(W)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=2, title="served routes")
    served, blocked = len(result.assignments), len(result.blocked)
    ax.set_title(title or f"Abilene lightpaths, hour {result.hour}: K={result.K}, W={W}, "
                          f"{served} served, {blocked} blocked ({result.method})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_heatmap(result, links: dict, path: str, title: str = "") -> str:
    """Links (rows) x wavelengths (columns); a filled cell shows the priority rank of the request using that
    wavelength on that link. A cell can hold only one request: asserted here as a visual proof of validity."""
    W = result.W
    link_ids = sorted(links)
    row = {l: i for i, l in enumerate(link_ids)}
    rank = {r.id: i + 1 for i, r in enumerate(result.requests)}          # 1 = highest predicted demand
    grid = np.zeros((len(link_ids), W), dtype=int)
    for lp in result.lightpaths:
        route = result.routes_by_request[lp["id"]]
        # recover the links of the chosen route from the node sequence
        for a, b in zip(lp["route"], lp["route"][1:]):
            i = row[tuple(sorted((a, b)))]
            assert grid[i, lp["wavelength"]] == 0, "wavelength clash on a link"
            grid[i, lp["wavelength"]] = rank[lp["id"]]
    fig, ax = plt.subplots(figsize=(max(6.4, 1.6 + 0.9 * W), 0.9 + 0.38 * len(link_ids)))
    ax.imshow(grid > 0, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(W):
            if grid[i, j]:
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(W)); ax.set_xticklabels([f"w{j}" for j in range(W)])
    ax.set_yticks(range(len(link_ids)))
    ax.set_yticklabels([f"{SHORT(a)}-{SHORT(b)}  ({(grid[i] > 0).sum()}/{W})" for i, (a, b) in enumerate(link_ids)],
                       fontsize=8)
    ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.set_yticks(np.arange(-.5, len(link_ids), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    ax.set_title(title or f"Wavelength use per link, hour {result.hour}, K={result.K}, W={W}\n"
                         "cell = priority rank of the request; (n/W) = wavelengths in use", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_nn_effect(data: dict, path: str) -> str:
    """Bar chart NN vs last-hour baseline inside the planner, one panel per (K, W) setting in `data`
    (written by eval.nn_effect). Error bars: std over seed means if >= 2 seeds, else standard error over hours."""
    settings = data["settings"]
    metrics = [("served_share", "served share"), ("blocking_rate", "blocking rate"), ("topk_hit_rate", "top-K hit rate")]
    fig, axes = plt.subplots(1, len(settings), figsize=(4.6 * len(settings), 4), squeeze=False)
    for ax, st in zip(axes[0], settings):
        rows = [r for r in data["runs"] if r["K"] == st["K"] and r["W"] == st["W"]]
        for k, (src, label, color) in enumerate((("nn", "NN forecast", "#2b6cb0"), ("last_hour", "last-hour", "#a0aec0"))):
            means, errs = [], []
            for key, _ in metrics:
                sel = [r for r in rows if r["source"] == src]
                seeds = sorted({r["seed"] for r in sel})
                per_seed = [np.mean([r[key] for r in sel if r["seed"] == s]) for s in seeds]
                means.append(float(np.mean(per_seed)))
                if len(seeds) >= 2:
                    errs.append(float(np.std(per_seed, ddof=1)))
                else:
                    vals = np.array([r[key] for r in sel]); errs.append(float(vals.std(ddof=1) / math.sqrt(len(vals))))
            x = np.arange(len(metrics)) + (k - 0.5) * 0.36
            ax.bar(x, means, 0.34, yerr=errs, capsize=3, label=label, color=color)
            for xi, m in zip(x, means):
                ax.text(xi, m + 0.01, f"{m:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m[1] for m in metrics], fontsize=8)
        ax.set_ylim(0, 1.08); ax.set_title(f"K={st['K']}, W={st['W']}  ({st['n_hours']} test hours, {st['n_seeds']} NN seed(s))", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    err_note = "error bars: std over NN seeds" if min(s["n_seeds"] for s in settings) >= 2 else "error bars: s.e. over test hours (1 seed)"
    fig.suptitle(f"Planner with NN forecast vs last-hour forecast ({err_note})", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def load_nn_effect(path: str):
    return json.load(open(path)) if os.path.exists(path) else None
