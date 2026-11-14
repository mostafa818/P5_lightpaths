"""Stage 6.2 plots (Member A): (a) topology + routes coloured by wavelength, (b) link x wavelength heatmap,
(d) NN on vs off. Plot (c), CSP domain sizes, is Member B's (src/eval/plot_csp.py)."""
from __future__ import annotations
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.common_eval import style, new_fig, INK

# Okabe-Ito: colour-blind-safe, 8 distinguishable colours (fits W = 8); tab20 beyond that
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#222222"]
NN_COLOR, LH_COLOR = "#2a78d6", "#9b9a96"

SHORT = lambda n: n.replace("ng", "").replace("M5", "-M5")      # ATLAng -> ATLA, ATLAM5 -> ATLA-M5


def _colors(W: int):
    if W <= len(OKABE_ITO):
        return OKABE_ITO[:W]
    cmap = plt.get_cmap("tab20")
    return [cmap(i % cmap.N) for i in range(W)]


def _text_color(hex_or_rgba) -> str:
    r, g, b = matplotlib.colors.to_rgb(hex_or_rgba)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "white"


def _xy(nodes: dict, n: str, lat0: float = 38.0):
    lon, lat = nodes[n]
    return lon * math.cos(math.radians(lat0)), lat


def plot_topology(result, nodes: dict, links: dict, path: str, title: str = "") -> str:
    """Topology at real coordinates; every served route drawn in the colour of its wavelength. Routes that
    share a link have different wavelengths, so each wavelength gets its own small lateral offset."""
    W = result.W
    cols = _colors(W)
    fig, ax = new_fig(figsize=(12, 6.6))
    for (u, v) in links:
        (x1, y1), (x2, y2) = _xy(nodes, u), _xy(nodes, v)
        ax.plot([x1, x2], [y1, y2], color="#dcdcd8", lw=2.5, solid_capstyle="round", zorder=1)
    step = 0.17                      # lateral distance between neighbouring wavelengths (about 3 line widths)
    for lp in result.lightpaths:
        off = (lp["wavelength"] - (W - 1) / 2) * step
        pts = []
        for a, b in zip(lp["route"], lp["route"][1:]):
            u, v = sorted((a, b))                          # canonical link direction: a wavelength keeps the same
            (ux, uy), (vx, vy) = _xy(nodes, u), _xy(nodes, v)   # side of a link whichever way the route travels
            norm = math.hypot(vx - ux, vy - uy) or 1.0
            ox, oy = -(vy - uy) / norm * off, (vx - ux) / norm * off
            (ax_, ay_), (bx_, by_) = _xy(nodes, a), _xy(nodes, b)
            pts += [(ax_ + ox, ay_ + oy), (bx_ + ox, by_ + oy)]   # consecutive segments are joined by a short jog
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=cols[lp["wavelength"]], lw=1.3, zorder=2,
                solid_joinstyle="round")
    for n in nodes:
        x, y = _xy(nodes, n)
        ax.scatter([x], [y], s=70, color="white", edgecolor="black", zorder=3)
        ax.annotate(SHORT(n), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, zorder=4)
    handles = [plt.Line2D([0], [0], color=cols[w], lw=3, label=f"wavelength {w}") for w in range(W)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=2, title="served routes", frameon=False)
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
        # the links of the chosen route come from its node sequence
        for a, b in zip(lp["route"], lp["route"][1:]):
            i = row[tuple(sorted((a, b)))]
            assert grid[i, lp["wavelength"]] == 0, "wavelength clash on a link"
            grid[i, lp["wavelength"]] = rank[lp["id"]]
    fig, ax = new_fig(figsize=(max(6.4, 1.6 + 0.9 * W), 0.9 + 0.38 * len(link_ids)))
    cols = _colors(W)
    rgb = np.ones((len(link_ids), W, 3))
    for i in range(grid.shape[0]):
        for j in range(W):
            if grid[i, j]:
                rgb[i, j] = matplotlib.colors.to_rgb(cols[j])
    ax.imshow(rgb, aspect="auto")                       # same colour per wavelength as the topology plot
    for i in range(grid.shape[0]):
        for j in range(W):
            if grid[i, j]:
                ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=8, color=_text_color(cols[j]))
    ax.set_xticks(range(W)); ax.set_xticklabels([f"w{j}" for j in range(W)])
    for j, lab in enumerate(ax.get_xticklabels()):
        lab.set_color(INK); lab.set_fontweight("bold")
    ax.set_yticks(range(len(link_ids)))
    ax.set_yticklabels([f"{SHORT(a)}-{SHORT(b)}  ({(grid[i] > 0).sum()}/{W})" for i, (a, b) in enumerate(link_ids)],
                       fontsize=8)
    ax.set_xticks(np.arange(-.5, W, 1), minor=True); ax.set_yticks(np.arange(-.5, len(link_ids), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title or f"Wavelength use per link, hour {result.hour}, K={result.K}, W={W}\n"
                         "cell = priority rank of the request; (n/W) = wavelengths in use", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


METRICS = [("served_share", "served share"), ("blocking_rate", "blocking rate"), ("topk_hit_rate", "top-K hit rate")]


def paired_differences(rows: list) -> dict:
    """Per-hour paired difference NN minus last-hour for each metric: {metric: (mean, SE, n_hours)}.
    With several NN seeds the per-hour difference is first averaged over the seeds; SE is over hours."""
    out = {}
    hours = sorted({r["hour"] for r in rows})
    seeds = sorted({r["seed"] for r in rows if r["source"] == "nn"})
    for key, _ in METRICS:
        lh = {r["hour"]: r[key] for r in rows if r["source"] == "last_hour"}
        nn = {(r["seed"], r["hour"]): r[key] for r in rows if r["source"] == "nn"}
        d = np.array([np.mean([nn[(s, h)] for s in seeds]) - lh[h] for h in hours])
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
        out[key] = (float(d.mean()), se, len(d))
    return out


def plot_nn_effect(data: dict, path: str) -> str:
    """Top row: NN vs last-hour inside the planner (one panel per (K, W) setting in `data`, written by
    eval.nn_effect); error bars: std over seed means if >= 2 seeds, else standard error over hours.
    Bottom row: paired per-hour difference NN minus last-hour with +-1 SE over hours (the more informative view)."""
    settings = data["settings"]
    n = len(settings)
    fig, axes = new_fig(2, n, figsize=(4.8 * n, 7.4), squeeze=False)
    for col, st in enumerate(settings):
        ax, ax2 = axes[0][col], axes[1][col]
        rows = [r for r in data["runs"] if r["K"] == st["K"] and r["W"] == st["W"]]
        for k, (src, label, color) in enumerate((("nn", "NN forecast", NN_COLOR), ("last_hour", "last-hour", LH_COLOR))):
            means, errs = [], []
            for key, _ in METRICS:
                sel = [r for r in rows if r["source"] == src]
                seeds = sorted({r["seed"] for r in sel})
                per_seed = [np.mean([r[key] for r in sel if r["seed"] == s]) for s in seeds]
                means.append(float(np.mean(per_seed)))
                if len(seeds) >= 2:
                    errs.append(float(np.std(per_seed, ddof=1)))
                else:
                    vals = np.array([r[key] for r in sel]); errs.append(float(vals.std(ddof=1) / math.sqrt(len(vals))))
            x = np.arange(len(METRICS)) + (k - 0.5) * 0.36
            ax.bar(x, means, 0.34, yerr=errs, capsize=3, label=label, color=color, ecolor=INK)
            for xi, m in zip(x, means):
                ax.text(xi, m + 0.03, f"{m:.2f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([m[1] for m in METRICS], fontsize=8)
        ax.set_ylim(0, 1.3)
        style(ax, ylabel="fraction", title=f"K={st['K']}, W={st['W']} ({st['n_hours']} test hours, {st['n_seeds']} NN seed(s))")
        ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=False)
        pd = paired_differences(rows)
        vals = [pd[k][0] for k, _ in METRICS]; ses = [pd[k][1] for k, _ in METRICS]
        ax2.bar(range(len(METRICS)), vals, 0.5, yerr=ses, capsize=4, color=NN_COLOR, ecolor=INK)
        ax2.axhline(0, color=INK, lw=0.8)
        for i, (v, s) in enumerate(zip(vals, ses)):
            ax2.text(i, v + (s + 0.0006 if v >= 0 else -s - 0.0006), f"{v:+.4f}\n±{s:.4f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7, color=INK)
        lim = max(0.012, 1.8 * max(abs(v) + s for v, s in zip(vals, ses)))
        ax2.set_ylim(-lim, lim)
        ax2.set_xticks(range(len(METRICS))); ax2.set_xticklabels([m[1] for m in METRICS], fontsize=8)
        style(ax2, ylabel="NN minus last-hour", title="paired difference per hour (±1 s.e.)")
    err_note = "top error bars: std over NN seeds" if min(s["n_seeds"] for s in settings) >= 2 else "top error bars: s.e. over test hours (1 seed)"
    fig.suptitle(f"Planner with NN forecast vs last-hour forecast ({err_note})", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def load_nn_effect(path: str):
    return json.load(open(path)) if os.path.exists(path) else None
