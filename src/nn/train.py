"""python -m nn.train : train 5 seeds, evaluate on the TEST week, save models + results.
Hyper-parameters are fixed in config.yaml (tuning, if any, only on validation)."""
import os, sys, json
import numpy as np, yaml, joblib
from nn.features import Scaler, build_dataset
from nn.model import train_mlp
from nn import evaluate as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    with open(os.path.join(ROOT, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    with open(os.path.join(ROOT, "data/processed/meta.json")) as fh:
        meta = json.load(fh)
    H = np.load(os.path.join(ROOT, "data/processed/hourly.npz"))["hourly"]
    (tr0, tr1), (va0, va1), (te0, te1) = (meta["split"][k] for k in ("train", "val", "test"))
    first = meta["split"]["first_target_hour"]
    train_ts, val_ts, test_ts = range(first, va0), range(va0, va1 + 1), range(te0, te1 + 1)

    scaler = Scaler.fit(H[:va0])                  # training hours only (0..431), not val/test
    Xtr, ytr = build_dataset(H, train_ts, scaler)
    Xva, yva = build_dataset(H, val_ts, scaler)
    Xte, yte = build_dataset(H, test_ts, scaler)
    ztr, zva = scaler.transform(ytr), scaler.transform(yva)
    print(f"rows: train {len(ytr)}, val {len(yva)}, test {len(yte)}; features {Xtr.shape[1]}")

    out_dir = os.path.join(ROOT, "outputs", "nn"); os.makedirs(out_dir, exist_ok=True)
    scaler.to_json(os.path.join(out_dir, "scaler.json"))
    true = H[list(test_ts)]
    base_pred = E.baseline_last_hour(H, test_ts)
    paired = {k: [] for k in E.KS}        # per-hour hit(NN) - hit(last hour), one array per seed
    results = {"baseline_last_hour": E.metrics(true, E.baseline_last_hour(H, test_ts)),
               "baseline_last_week": E.metrics(true, E.baseline_last_week(H, test_ts)),
               "val_baseline_last_hour": E.metrics(H[list(val_ts)], E.baseline_last_hour(H, val_ts)),
               "nn_seeds": {}}
    n = cfg["nn"]
    for seed in n["seeds"]:
        m, hist = train_mlp(seed, Xtr, ztr, Xva, zva, n["hidden"], n["lr"], n["batch_size"],
                            n["max_epochs"], n["patience"])
        joblib.dump(m, os.path.join(out_dir, f"mlp_seed{seed}.joblib"))
        pred = scaler.inverse(m.predict(Xte)).reshape(len(test_ts), -1)
        vpred = scaler.inverse(m.predict(Xva)).reshape(len(val_ts), -1)
        for k in E.KS:
            paired[k].append(E.topk_hits_per_hour(true, pred, k) - E.topk_hits_per_hour(true, base_pred, k))
        r = E.metrics(true, pred); r["epochs"] = len(hist)
        r["best_val_mse_log"] = min(h["val_mse"] for h in hist)
        r["val"] = E.metrics(H[list(val_ts)], vpred)
        results["nn_seeds"][str(seed)] = r
        with open(os.path.join(out_dir, f"curve_seed{seed}.json"), "w") as fh:
            json.dump(hist, fh)
        print(f"seed {seed}: epochs {r['epochs']} " + " ".join(f"{k} {v:.3f}" for k, v in r.items() if k in ("mae", "rmse", "hit@10", "hit@20", "hit@40")))
    keys = ["mae", "rmse", "hit@10", "hit@20", "hit@40"]
    results["nn_mean"] = {k: float(np.mean([r[k] for r in results["nn_seeds"].values()])) for k in keys}
    results["nn_std"] = {k: float(np.std([r[k] for r in results["nn_seeds"].values()], ddof=1)) for k in keys}
    # paired difference vs last-hour per test hour (seed-averaged); SE over the 168 test hours
    results["paired_hit_diff_vs_last_hour"] = {}
    for k in E.KS:
        d = np.mean(paired[k], axis=0)
        results["paired_hit_diff_vs_last_hour"][f"hit@{k}"] = {
            "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n_hours": int(len(d))}
    with open(os.path.join(ROOT, "outputs", "nn_eval.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("\n%-22s" % "method" + " ".join("%10s" % k for k in keys))
    for name in ("baseline_last_hour", "baseline_last_week"):
        print("%-22s" % name + " ".join("%10.3f" % results[name][k] for k in keys))
    print("%-22s" % "NN mean" + " ".join("%10.3f" % results["nn_mean"][k] for k in keys))
    for k, v in results["paired_hit_diff_vs_last_hour"].items():
        print(f"paired {k} NN - last hour: {v['mean']:+.4f} +- {v['se']:.4f} (SE)")
    print("%-22s" % "NN std" + " ".join("%10.3f" % results["nn_std"][k] for k in keys))


if __name__ == "__main__":
    main()
