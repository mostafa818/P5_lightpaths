"""Shared helpers for the experiments: paths, plot style, markdown tables, mean/std formatting."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "experiments"
with open(ROOT / "config.yaml") as _fh:
    CFG = yaml.safe_load(_fh)

# categorical slots 1-3 of the reference palette (blue, orange, aqua): fixed order, one colour per method
COLORS = {"BFS": "#2a78d6", "UCS": "#eb6834", "A*": "#1baf7a",
          "BT": "#2a78d6", "BT+FC": "#eb6834", "BT+FC+AC3": "#1baf7a"}
INK, GRID = "#0b0b0b", "#e3e2de"


def style(ax, ylabel=None, title=None):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color("#52514e")
    ax.tick_params(colors="#52514e")
    if ylabel:
        ax.set_ylabel(ylabel, color="#52514e")
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")


def new_fig(*a, **k):
    fig, axes = plt.subplots(*a, **k)
    fig.patch.set_facecolor("#fcfcfb")
    return fig, axes


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def pm(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def md_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"
