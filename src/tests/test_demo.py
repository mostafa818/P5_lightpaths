"""Stage 6 smoke tests (Member A): run_demo.py, plots (a), (b), (d) and the NN-effect harness."""
import importlib.util, json, os, sys
import numpy as np, pytest
import matplotlib.image as mpimg

from integration.pipeline import Planner, plan
from search.graph import Graph
from eval import plots

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOURLY = os.path.join(ROOT, "data", "processed", "hourly.npz")
MODEL = os.path.join(ROOT, "outputs", "nn", "mlp_seed0.joblib")
TOPO = os.path.join(ROOT, "data", "raw", "topology", "abilene_sndlib_native.txt")
needs_abilene = pytest.mark.skipif(not (os.path.exists(HOURLY) and os.path.exists(MODEL) and os.path.exists(TOPO)),
                                   reason="needs processed data, NN models and the Abilene topology file")


def run_demo(argv, capsys):
    spec = importlib.util.spec_from_file_location("run_demo", os.path.join(ROOT, "run_demo.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    old = sys.argv
    sys.argv = ["run_demo.py"] + argv
    try:
        mod.main()
    finally:
        sys.argv = old
    return capsys.readouterr().out


# ---------- toy result for the plot functions ----------------------------------------------------------
NODES = {"S": (-100.0, 40.0), "A": (-98.0, 41.0), "B": (-98.0, 39.0), "D": (-96.0, 40.0)}
EDGES = [("S", "A", 10.0), ("A", "D", 10.0), ("S", "B", 12.0), ("B", "D", 12.0)]
LINKS = {tuple(sorted((u, v))): L for u, v, L in EDGES}


def toy_result(W=2, n=3):
    pl = Planner(Graph.from_edges(EDGES), [("S", "D")] * n, 100.0)
    return plan(pl, 0, np.arange(n, 0, -1, dtype=float), K=n, W=W)


def valid_png(path):
    assert os.path.getsize(path) > 2000
    img = mpimg.imread(path)
    assert img.ndim == 3 and img.shape[0] > 100 and img.shape[1] > 100


def test_topology_and_heatmap_write_images(tmp_path):
    r = toy_result(W=2, n=3)                              # 3 requests, 2 routes, W=2 -> some are served on both routes
    assert len(r.assignments) >= 2
    valid_png(plots.plot_topology(r, NODES, LINKS, str(tmp_path / "t.png")))
    valid_png(plots.plot_heatmap(r, LINKS, str(tmp_path / "h.png")))


def test_plots_handle_empty_result(tmp_path):
    r = toy_result(W=2, n=3)
    r.assignments, r.routes_by_request = {}, {}
    valid_png(plots.plot_topology(r, NODES, LINKS, str(tmp_path / "t.png")))
    valid_png(plots.plot_heatmap(r, LINKS, str(tmp_path / "h.png")))


def test_heatmap_detects_a_wavelength_clash(tmp_path):
    r = toy_result(W=2, n=2)
    ids = sorted(r.assignments)
    r.assignments = {ids[0]: (0, 0), ids[1]: (0, 0)}      # both on route 0, same wavelength -> clash
    with pytest.raises(AssertionError):
        plots.plot_heatmap(r, LINKS, str(tmp_path / "h.png"))


def test_nn_effect_plot_from_synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    runs = []
    for W in (4, 8):
        for src, seed in (("last_hour", -1), ("nn", 0), ("nn", 1)):
            for t in range(10):
                runs.append({"source": src, "seed": seed, "hour": t, "K": 20, "W": W, "served_share": rng.random(),
                             "blocking_rate": rng.random(), "topk_hit_rate": rng.random()})
    data = {"settings": [{"K": 20, "W": 4, "n_hours": 10, "n_seeds": 2}, {"K": 20, "W": 8, "n_hours": 10, "n_seeds": 2}],
            "runs": runs}
    valid_png(plots.plot_nn_effect(data, str(tmp_path / "n.png")))
    assert plots.load_nn_effect(str(tmp_path / "missing.json")) is None


# ---------- run_demo.py ---------------------------------------------------------------------------------
@needs_abilene
def test_demo_clean_run(capsys):
    out = run_demo([], capsys)                             # hour 540, K=20, W=8, FC+AC-3, NN on
    for section in ("=== INPUT", "=== FEEDBACK LOOP", "=== ROUTES", "=== ASSIGNMENT TABLE", "=== BLOCKED (0)", "=== SUMMARY"):
        assert section in out
    assert "served 20/20" in out and "NN (seed 0)" in out


@needs_abilene
def test_demo_feedback_loop_run(capsys):
    out = run_demo(["--W", "4"], capsys)
    assert "served 8/20" in out and "wavelengths exhausted" in out and "[more routes]" in out


@needs_abilene
def test_demo_nn_off_and_methods(capsys):
    off = run_demo(["--nn", "off", "--K", "10"], capsys)
    assert "last hour (t-1)" in off and "served 10/10" in off
    for m in ("bt", "fc", "fc+ac3"):
        assert "served 10/10" in run_demo(["--method", m, "--K", "10"], capsys)


@needs_abilene
def test_demo_strict_mode(capsys):
    out = run_demo(["--W", "4", "--strict"], capsys)
    assert "status: no_solution" in out and "No solution: wavelengths exhausted" in out and "served 0/20" in out


@needs_abilene
def test_demo_rejects_hour_without_history():
    with pytest.raises(SystemExit):
        spec = importlib.util.spec_from_file_location("run_demo", os.path.join(ROOT, "run_demo.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        old = sys.argv; sys.argv = ["run_demo.py", "--hour", "100"]
        try:
            mod.main()
        finally:
            sys.argv = old


@needs_abilene
def test_demo_plots_and_trace_hook(capsys, tmp_path):
    out = run_demo(["--W", "4", "--plots", "--trace", "--outdir", str(tmp_path)], capsys)
    assert "=== SEARCH TRACE" in out and "=== FIGURES" in out
    for f in ("demo_topology_h540_K20_W4.png", "demo_heatmap_h540_K20_W4.png"):
        valid_png(str(tmp_path / f))


@needs_abilene
def test_nn_effect_harness_small_run():
    from eval.nn_effect import run
    rows = run(K=10, W=8, seeds=[0], hours=[510, 520])
    assert len(rows) == 4 and {r["source"] for r in rows} == {"nn", "last_hour"}
    for r in rows:
        assert 0 <= r["served_share"] <= 1 and 0 <= r["blocking_rate"] <= 1 and 0 <= r["topk_hit_rate"] <= 1


def test_paired_differences_hand_computed():
    rows = []
    for h, (nn, lh) in enumerate([(0.9, 0.8), (0.7, 0.8), (0.8, 0.6)]):          # per-hour served share
        rows.append({"source": "last_hour", "seed": -1, "hour": h, "served_share": lh, "blocking_rate": 0.0, "topk_hit_rate": 1.0})
        for s in (0, 1):
            rows.append({"source": "nn", "seed": s, "hour": h, "served_share": nn, "blocking_rate": 0.0, "topk_hit_rate": 1.0})
    pd = plots.paired_differences(rows)
    diffs = np.array([0.1, -0.1, 0.2])
    mean, se, n = pd["served_share"]
    assert n == 3 and mean == pytest.approx(diffs.mean()) and se == pytest.approx(diffs.std(ddof=1) / np.sqrt(3))
    assert pd["blocking_rate"] == (0.0, 0.0, 3)


def test_wavelength_palette_has_distinct_colours_up_to_eight():
    assert len(set(plots._colors(8))) == 8 and len(set(plots._colors(12))) == 12
