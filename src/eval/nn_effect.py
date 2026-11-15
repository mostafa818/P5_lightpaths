"""NN vs last-hour forecast inside the planner (feeds plot (d) now and Experiment 3 later).

python -m eval.nn_effect --K 20 --W 4 8 --seeds 0 --out outputs/nn_effect.json
Experiment 3 (plan 7.3) is this function with seeds 0-4 and the full test week."""
import argparse, json, os, time
import numpy as np
from integration.pipeline import default_planner, plan, evaluate, nn_forecast_fn, last_hour_forecast_fn


def run(K: int, W: int, seeds, hours, method: str = "fc+ac3", planner=None, H=None, max_assignments=None,
        include_last_hour: bool = True):
    if planner is None:
        planner, H, cfg = default_planner()
        max_assignments = max_assignments or cfg["csp"]["max_assignments"]
    rows = []
    sources = ([("last_hour", None, last_hour_forecast_fn(H))] if include_last_hour else []) + \
              [("nn", s, nn_forecast_fn(s)) for s in seeds]
    for src, seed, fn in sources:
        for t in hours:
            r = plan(planner, t, fn(t), K, W, method=method, max_assignments=max_assignments)
            m = evaluate(r, H[t])
            rows.append({"source": src, "seed": -1 if seed is None else seed, "hour": int(t), "K": K, "W": W, **m})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20); ap.add_argument("--W", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0]); ap.add_argument("--method", default="fc+ac3")
    ap.add_argument("--out", default="outputs/nn_effect.json")
    a = ap.parse_args()
    planner, H, cfg = default_planner()
    hours = range(cfg["split"]["test_hours"][0], cfg["split"]["test_hours"][1] + 1)
    out = {"settings": [], "runs": []}
    for W in a.W:
        t0 = time.time()
        rows = run(a.K, W, a.seeds, hours, a.method, planner, H, cfg["csp"]["max_assignments"])
        out["runs"] += rows
        out["settings"].append({"K": a.K, "W": W, "n_hours": len(hours), "n_seeds": len(a.seeds), "method": a.method})
        for src in ("last_hour", "nn"):
            sel = [r for r in rows if r["source"] == src]
            print(f"K={a.K} W={W} {src:9s} served {np.mean([r['served_share'] for r in sel]):.3f} "
                  f"blocking {np.mean([r['blocking_rate'] for r in sel]):.3f} hit {np.mean([r['topk_hit_rate'] for r in sel]):.3f}")
        from eval.plots import paired_differences
        pd_ = paired_differences(rows)
        print("  paired NN - last hour (mean +- s.e. over hours): " +
              ", ".join(f"{k} {m:+.4f} +- {s:.4f}" for k, (m, s, _) in pd_.items()))
        print(f"  ({time.time() - t0:.0f}s)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
