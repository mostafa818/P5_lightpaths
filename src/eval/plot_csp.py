"""Plot (c), plan 6.2 (Member B): CSP domain sizes with and without AC-3 and forward checking.

For each request (x axis, rank 1 = highest predicted demand) three bars: the initial domain (routes x W), the domain
after AC-3, and the domain after forward checking following the first assignments (see ``csp.analysis``). One panel
per instance. ``python -m eval.plot_csp`` writes ``outputs/csp_domains.png`` for the two demo instances agreed in
issue #7: hour 540, K=20, W=4 (forward checking shrinks the domains; AC-3 prunes nothing) and hour 540, K=10, W=1
(AC-3 empties a domain and proves the instance unsolvable before any search).
"""
from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

from csp.analysis import domain_sizes
from eval.common_eval import COLORS, ROOT, new_fig, out_dir, style

BARS = [("initial", "initial (routes x W)", "#9b9a96"),
        ("after_ac3", "after AC-3", COLORS["BT+FC+AC3"]),
        ("after_fc", "after forward checking", COLORS["BT+FC"])]


def _draw(ax, title: str, s: dict) -> None:
    n = len(s["requests"])
    x = np.arange(n)
    for k, (key, label, color) in enumerate(BARS):
        vals = s[key]
        if vals is None:
            continue
        ax.bar(x + (k - 1) * 0.28, vals, width=0.26, color=color, label=label)
    ax.set_xticks(x, [str(i + 1) for i in range(n)], fontsize=7)
    ax.set_xlabel("request (rank by predicted demand)", color="#52514e")
    style(ax, ylabel="values left in the domain", title=title)
    ax.set_title(title, color="#0b0b0b", fontsize=10, loc="left", pad=34)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    notes = [f"total values: initial {sum(s['initial'])}, after AC-3 {sum(s['after_ac3'])}"
             + (f", after FC {sum(s['after_fc'])}" if s["after_fc"] is not None else "")]
    if s["ac3_wiped_out"]:
        notes.append(f"AC-3 empties the domain of the request ranked {s['requests'].index(s['wiped_out_request']) + 1}: "
                     "no solution, 0 assignments")
    elif sum(s["after_ac3"]) == sum(s["initial"]):
        notes.append("AC-3 removes nothing here (every value has support on another wavelength)")
    if s["after_fc"] is not None:
        notes.append(f"FC: after the first {s['fc_steps']} assignment(s); assigned requests show 1")
    ax.text(0, 1.02, "   ".join(notes[:1]) + "\n" + "   ".join(notes[1:]), transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7.5, color="#52514e")


def make_plot(panels, path) -> str:
    """``panels``: one ``domain_sizes`` dict, or a list of ``(title, dict)``. Writes the PNG and returns the path."""
    if isinstance(panels, dict):
        panels = [("CSP domain sizes", panels)]
    fig, axes = new_fig(len(panels), 1, figsize=(11, 3.6 * len(panels)), squeeze=False)
    for ax, (title, s) in zip(axes[:, 0], panels):
        _draw(ax, title, s)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    return str(path)


def demo_panels(hour: int = 540) -> list:
    """The two demo instances (NN forecast): data needed, see the module docstring."""
    from eval.exp2_csp import default_planner, instance, nn_forecast_fn
    planner, _, _ = default_planner()
    pred = nn_forecast_fn(0)(hour)
    out = []
    for K, W, steps in ((20, 4, 3), (10, 1, 1)):
        reqs, routes = instance(planner, hour, K, pred)
        out.append((f"hour {hour}, K={K}, W={W}", domain_sizes(reqs, routes, W, fc_steps=steps)))
    return out


def main() -> None:
    path = out_dir().parent / "csp_domains.png"
    print(make_plot(demo_panels(), path))


if __name__ == "__main__":
    main()
