"""Experiment 3 (Member A, plan 7.3): does a better forecast help the planner?

For each of the 5 NN seeds and every test hour (504-671) the Stage 5 pipeline runs with the NN forecast and with
the last-hour forecast, at K = 20 and W = 4 (the plan's example; blocking dominates) and W = 8 (almost everything
fits). Metrics: served share, blocking rate, top-K hit rate (plan 5.6, true demand of the same hour).

python -m eval.exp3_nn                  # full run (~6 min, resumable: finished chunks are kept)
python -m eval.exp3_nn --reuse          # rebuild tables/plot from the saved CSV
python -m eval.exp3_nn --max-seconds N  # stop after about N s; rerun to continue
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from eval.common_eval import CFG, ROOT, md_table, out_dir, pm
from eval.nn_effect import run as run_planner
from eval.plots import METRICS, plot_nn_effect
from integration.pipeline import default_planner

SETTINGS = [(20, 4), (20, 8)]                                  # (K, W)
SEEDS = CFG["nn"]["seeds"]                                     # 5 NN seeds
HOURS = list(range(CFG["split"]["test_hours"][0], CFG["split"]["test_hours"][1] + 1))
BLOCK, N_BOOT = 24, 2000                                       # moving-block bootstrap: one-day blocks
BETTER = {"served_share": +1, "blocking_rate": -1, "topk_hit_rate": +1}   # sign of an improvement


def run(max_seconds: float = None):
    """All planner runs, one chunk per (K, W, forecast source / NN seed). Finished chunks are kept in
    outputs/experiments/.exp3_chunks/ and skipped on a rerun, so an interrupted run resumes. Returns the full
    DataFrame, or None if `max_seconds` ran out first (just run the command again)."""
    planner, H, cfg = default_planner()
    chunk_dir = out_dir() / ".exp3_chunks"
    chunk_dir.mkdir(exist_ok=True)
    t0, frames = time.time(), []
    for K, W in SETTINGS:
        for src, seed in [("last_hour", None)] + [("nn", s) for s in SEEDS]:
            f = chunk_dir / f"K{K}_W{W}_{src}{'' if seed is None else seed}.csv"
            if not f.exists():
                if max_seconds is not None and time.time() - t0 > max_seconds:
                    print(f"  stopped after {time.time() - t0:.0f}s; run the same command again to continue", flush=True)
                    return None
                t1 = time.time()
                rows = run_planner(K, W, [] if seed is None else [seed], HOURS, "fc+ac3", planner, H,
                                   cfg["csp"]["max_assignments"], include_last_hour=seed is None)
                pd.DataFrame(rows).to_csv(f, index=False)
                print(f"  K={K} W={W} {src}{'' if seed is None else ' seed ' + str(seed)}: {len(rows)} runs ({time.time() - t1:.0f}s)", flush=True)
            frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def paired_hourly(df: pd.DataFrame, K: int, W: int, key: str) -> pd.Series:
    """Per-hour NN minus last-hour; the NN value is the mean over seeds for that hour."""
    d = df[(df.K == K) & (df.W == W)]
    lh = d[d.source == "last_hour"].set_index("hour")[key]
    nn = d[d.source == "nn"].groupby("hour")[key].mean()
    return (nn - lh).sort_index()


def block_bootstrap_ci(diffs: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = None, level=0.95):
    """CI for the mean of an autocorrelated hourly series: resample whole blocks of `block` consecutive hours."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(CFG["seed"] if seed is None else seed)
    block = min(block, n)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(starts, size=k)
        means[b] = np.concatenate([diffs[s:s + block] for s in pick])[:n].mean()
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def summarise_setting(df: pd.DataFrame, K: int, W: int) -> tuple:
    """(markdown table, per-seed paired mean differences) for one (K, W)."""
    d = df[(df.K == K) & (df.W == W)]
    seeds = sorted(d[d.source == "nn"].seed.unique())
    rows, per_seed = [], {}
    for key, label in METRICS:
        lh = d[d.source == "last_hour"].set_index("hour")[key]
        nn_seed_means = [d[(d.source == "nn") & (d.seed == s)][key].mean() for s in seeds]
        diff = paired_hourly(df, K, W, key)
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        lo, hi = block_bootstrap_ci(diff.to_numpy())
        sign = BETTER[key]
        better, worse = int((sign * diff > 1e-12).sum()), int((sign * diff < -1e-12).sum())
        per_seed[key] = [float((d[(d.source == "nn") & (d.seed == s)].set_index("hour")[key] - lh).mean()) for s in seeds]
        rows.append([label, f"{lh.mean():.3f}", pm(float(np.mean(nn_seed_means)), float(np.std(nn_seed_means, ddof=1)), 3),
                     f"{diff.mean():+.4f} ± {se:.4f}", f"[{lo:+.4f}, {hi:+.4f}]",
                     f"{better} / {len(diff) - better - worse} / {worse}"])
    table = md_table(["metric", "last-hour", f"NN (mean ± std over {len(seeds)} seeds)", "NN − last-hour, paired (mean ± s.e. over hours)",
                      "95% block-bootstrap CI", "hours NN better / equal / worse"], rows)
    return table, per_seed


def write_outputs(df: pd.DataFrame, od, json_path=None, write_json: bool = True) -> None:
    parts = []
    for K, W in SETTINGS:
        table, per_seed = summarise_setting(df, K, W)
        spread = "; ".join(f"{label}: " + ", ".join(f"{v:+.4f}" for v in per_seed[key]) for key, label in METRICS)
        parts.append(f"### K={K}, W={W}\n\n{table}\nPer-seed paired mean difference NN − last-hour ({spread}).\n")
    text = ("## Experiment 3: NN vs last-hour forecast inside the planner\n\n"
            f"{len(HOURS)} test hours (504-671), {len(SEEDS)} NN seeds, planner BT+FC+AC-3 with precheck, cap "
            f"{CFG['csp']['max_assignments']:,} assignments, k = 3 routes (5 on retry), reach {CFG['rwa']['reach_km']} km. "
            "Served share and top-K hit rate: higher is better; blocking rate: lower is better. The last-hour "
            "forecast is deterministic, so it has no seed spread. The paired difference averages the 5 seeds per hour; its "
            "s.e. treats the 168 hours as independent, which they are not (hourly traffic is autocorrelated), so the "
            f"moving-block bootstrap CI ({BLOCK}-hour blocks, {N_BOOT} resamples, seed {CFG['seed']}) is the more "
            f"reliable interval; with only {len(HOURS) // BLOCK} blocks it is still rough. The top-K hit rate depends on the "
            "forecast only, not on W, so its two rows are one comparison; blocking at W=8 is zero for every run. That leaves "
            "four distinct comparisons (served share at W=4 and W=8, blocking at W=4, hit rate), so a borderline interval "
            "should be read with caution.\n\n" + "\n".join(parts))
    (od / "exp3_nn_summary.md").write_text(text)
    data = {"settings": [{"K": K, "W": W, "n_hours": len(HOURS), "n_seeds": len(SEEDS), "method": "fc+ac3"} for K, W in SETTINGS],
            "runs": df.to_dict("records")}
    if write_json:                                                                  # also feeds plot (d) of run_demo.py
        json.dump(data, open(json_path or ROOT / "outputs" / "nn_effect.json", "w"))
    plot_nn_effect(data, str(od / "exp3_nn.png"))
    print(text)


def main(reuse: bool = False, max_seconds: float = None) -> None:
    t0 = time.time()
    od = out_dir()
    if reuse:
        # --reuse leaves an existing outputs/nn_effect.json alone (a CSV round trip only changes float formatting)
        write_outputs(pd.read_csv(od / "exp3_nn_raw.csv"), od, write_json=not (ROOT / "outputs" / "nn_effect.json").exists())
        return
    df = run(max_seconds)
    if df is None:
        return
    df.to_csv(od / "exp3_nn_raw.csv", index=False)
    write_outputs(df, od)
    print(f"[exp3 done in {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ms = float(sys.argv[sys.argv.index("--max-seconds") + 1]) if "--max-seconds" in sys.argv else None
    main(reuse="--reuse" in sys.argv, max_seconds=ms)
