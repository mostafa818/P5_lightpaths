"""Steps 1.2-1.10: build data/processed/hourly.npz and meta.json (reproducible, seeded)."""
import sys, os, json, hashlib, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np, yaml
from data import pipeline as P

ROOT = os.path.dirname(os.path.abspath(__file__))
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
np.random.seed(cfg["seed"])   # this stage is deterministic (no random step); seeded for safety
raw_dir = os.path.join(ROOT, cfg["paths"]["raw"])
out_dir = os.path.join(ROOT, cfg["paths"]["processed"])
os.makedirs(out_dir, exist_ok=True)
nodes = cfg["node_order"]
assert len(nodes) == 12

weeks = ["X06", "X07", "X08", "X09"]   # 2004-05-01 .. 2004-05-28, consecutive weeks
missing = [w + ".gz" for w in weeks if not os.path.exists(os.path.join(raw_dir, w + ".gz"))]
if missing:
    sys.exit(f"Missing raw files in {raw_dir}: {missing}\n"
             "Download Yin Zhang's Abilene weekly files (X06-X09.gz) into data/raw/ - see README 'Getting the raw data'.")
hourly, files, week5min = [], {}, []
for w in weeks:
    path = os.path.join(raw_dir, w + ".gz")
    x = P.to_mbps(P.load_week_real(path))           # 1.2, 1.3, 1.5
    week5min.append(x.sum(axis=1))                  # total Mbps per 5-min slot (boundary check)
    h = P.drop_diagonal(P.to_hourly(x))             # 1.6, 1.7
    assert h.shape == (168, 132), h.shape
    hourly.append(h)
    files[w] = {"bytes": os.path.getsize(path), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
H = np.concatenate(hourly, axis=0)                  # 1.10: (672, 132)
chk = P.sanity_check(H)                             # 1.8
# 1.8 boundary continuity: compare median total of last vs first 12 five-minute slots (a burst
# inside one hour is real traffic, not a file-boundary jump; see docs/data_dictionary.md)
bnd = []
for k in range(len(weeks) - 1):
    a, b = np.median(week5min[k][-12:]), np.median(week5min[k + 1][:12])
    bnd.append(round(float(b / a), 3))
    assert 0.5 < b / a < 2.0, f"discontinuity between {weeks[k]} and {weeks[k+1]}: {b/a:.2f}"
chk["boundary_ratio_first_hour_over_last_hour_median"] = bnd
assert chk["n_nan"] == 0 and chk["n_negative"] == 0 and chk["n_allzero_rows"] == 0, chk

tr0, tr1 = cfg["split"]["train_hours"]; te0, te1 = cfg["split"]["test_hours"]
val0 = tr1 + 1 - 24 * cfg["split"]["val_last_days"]
pairs = P.pair_list(nodes)
meta = {
    "seed": cfg["seed"], "node_order": nodes, "pairs": pairs,
    "week_files": files, "week_start_dates": {"X06": "2004-05-01", "X07": "2004-05-08", "X08": "2004-05-15", "X09": "2004-05-22"},
    "start_datetime": "2004-05-01T00:00", "start_weekday_mon0": datetime.date(2004, 5, 1).weekday(),
    "unit": "Mbps (hourly mean)", "mbps_factor": P.MBPS_FACTOR,
    "split": {"train": [tr0, tr1], "val": [val0, tr1], "test": [te0, te1], "first_target_hour": 168},
    "sanity": chk, "shape": list(H.shape),
}
np.savez_compressed(os.path.join(out_dir, "hourly.npz"), hourly=H)
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(11, 6))
ax[0].plot(H.sum(axis=1), lw=0.8)
for b in (168, 336, 504): ax[0].axvline(b, color="r", ls="--", lw=0.8)
ax[0].axvline(504, color="g", lw=1.2); ax[0].set_title("Total Abilene traffic per hour, X06-X09 (red = week boundaries, green = train|test)")
ax[0].set_xlabel("hour"); ax[0].set_ylabel("Mbps")
ax[1].plot(H.sum(axis=1).reshape(-1, 24).mean(axis=0), marker="o"); ax[1].set_title("Mean total by hour of day (UTC)")
ax[1].set_xlabel("hour of day"); ax[1].set_ylabel("Mbps"); fig.tight_layout()
os.makedirs(os.path.join(ROOT, cfg["paths"]["outputs"]), exist_ok=True)
fig.savefig(os.path.join(ROOT, cfg["paths"]["outputs"], "total_traffic_hourly.png"), dpi=110)
print("saved", H.shape, "total Mbps/hour mean %.1f max %.1f" % (H.sum(1).mean(), H.sum(1).max()), chk)
