import os, json
from pathlib import Path
import numpy as np, pytest
from data import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOURLY = os.path.join(ROOT, "data", "processed", "hourly.npz")
RAW_X06 = os.path.join(ROOT, "data", "raw", "X06.gz")
needs_processed = pytest.mark.skipif(not os.path.exists(HOURLY), reason="run `python make_data.py` first")
needs_raw = pytest.mark.skipif(not os.path.exists(RAW_X06), reason="raw data not downloaded (git-ignored)")


def test_unit_factor():
    assert P.to_mbps(np.array([300e6 / 800]))[0] == pytest.approx(1.0)   # 100 B/5min*... = 1 Mbps


def test_to_hourly_is_mean_of_12():
    x = np.arange(24 * 3, dtype=float).reshape(24, 3)
    h = P.to_hourly(x)
    assert h.shape == (2, 3)
    assert np.allclose(h[0], x[:12].mean(axis=0)) and np.allclose(h[1], x[12:].mean(axis=0))


def test_drop_diagonal_and_pairs():
    x = np.arange(144, dtype=float)[None, :]
    d = P.drop_diagonal(x)
    assert d.shape == (1, 132)
    nodes = [f"n{i}" for i in range(12)]
    pairs = P.pair_list(nodes)
    assert len(pairs) == 132 and all(s != t for s, t in pairs)
    # element for pair (n0, n1) is flattened index 0*12+1 = 1; (n1, n0) is 12
    assert d[0, pairs.index(("n0", "n1"))] == 1 and d[0, pairs.index(("n1", "n0"))] == 12


def test_hourly_requires_multiple_of_12():
    with pytest.raises(ValueError):
        P.to_hourly(np.zeros((13, 144)))


@needs_processed
def test_processed_shape_and_values():
    H = np.load(HOURLY)["hourly"]
    assert H.shape == (672, 132)
    assert not np.isnan(H).any() and (H >= 0).all()
    assert (H.sum(axis=1) > 0).all()


@needs_processed
def test_meta_consistent():
    m = json.loads(Path(os.path.join(ROOT, "data", "processed", "meta.json")).read_text())
    assert len(m["pairs"]) == 132 and len(m["node_order"]) == 12
    assert m["split"]["train"] == [0, 503] and m["split"]["test"] == [504, 671]
    assert m["split"]["val"] == [432, 503] and m["start_weekday_mon0"] == 5   # 2004-05-01 = Saturday


@needs_processed
@needs_raw
def test_hourly_matches_manual_average():
    H = np.load(HOURLY)["hourly"]
    x = P.to_mbps(P.load_week_real(RAW_X06))          # (2016, 144) Mbps
    manual = x[12:24, 1 * 12 + 2].mean()               # hour 1, pair (router 1 -> router 2)
    m = json.loads(Path(os.path.join(ROOT, "data", "processed", "meta.json")).read_text())
    k = m["pairs"].index([m["node_order"][1], m["node_order"][2]])
    assert H[1, k] == pytest.approx(manual)


# ---------- Stage 8 additions (Member A) ----------------------------------------------------------------
def test_load_week_real_rejects_a_file_with_the_wrong_shape(tmp_path):
    import gzip
    bad = tmp_path / "bad.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError, match="unexpected shape"):
        P.load_week_real(str(bad))


def test_sanity_check_counts_problems():
    h = np.ones((4, 3))
    h[0, 0] = np.nan
    h[1, 1] = -2.0
    h[2] = 0.0                                # an all-zero hour
    h[:, 2] = 0.0                             # an all-zero pair
    h[3, 0], h[3, 1] = 1.0, -1.0              # a row and a column whose entries cancel to 0 are NOT all-zero
    s = P.sanity_check(h)
    assert s["n_nan"] == 1 and s["n_negative"] == 2 and s["n_allzero_rows"] == 1 and s["n_allzero_pairs"] == 1
    clean = P.sanity_check(np.ones((5, 4)))
    assert clean == {"n_nan": 0, "n_negative": 0, "n_allzero_rows": 0, "n_allzero_pairs": 0}


@needs_processed
@needs_raw
def test_committed_dataset_is_reproduced_from_the_raw_files_and_hashes_match():
    """Re-run the pipeline on X06-X09 and compare with the committed hourly.npz; raw files match their recorded SHA-256."""
    import hashlib
    meta = json.loads(Path(os.path.join(ROOT, "data", "processed", "meta.json")).read_text())
    parts = []
    for w, info in meta["week_files"].items():
        path = os.path.join(ROOT, "data", "raw", w + ".gz")
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == info["sha256"]
        parts.append(P.drop_diagonal(P.to_hourly(P.to_mbps(P.load_week_real(path)))))
    assert np.array_equal(np.concatenate(parts), np.load(HOURLY)["hourly"])


@needs_processed
@needs_raw
def test_make_data_script_reproduces_the_committed_files(tmp_path):
    """Run the real `make_data.py` in a scratch folder (raw files symlinked) and compare with the committed output."""
    import subprocess, sys, shutil
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir()
    for w in ("X06", "X07", "X08", "X09"):
        os.symlink(os.path.join(ROOT, "data", "raw", w + ".gz"), tmp_path / "data" / "raw" / (w + ".gz"))
    os.symlink(os.path.join(ROOT, "src"), tmp_path / "src")
    shutil.copy(os.path.join(ROOT, "make_data.py"), tmp_path / "make_data.py")
    shutil.copy(os.path.join(ROOT, "config.yaml"), tmp_path / "config.yaml")
    done = subprocess.run([sys.executable, "make_data.py"], cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr
    assert np.array_equal(np.load(tmp_path / "data" / "processed" / "hourly.npz")["hourly"], np.load(HOURLY)["hourly"])
    assert json.loads((tmp_path / "data" / "processed" / "meta.json").read_text()) == \
        json.loads(Path(os.path.join(ROOT, "data", "processed", "meta.json")).read_text())
    assert (tmp_path / "outputs" / "total_traffic_hourly.png").stat().st_size > 2000


@needs_raw
def test_make_data_fails_clearly_when_a_raw_file_is_missing(tmp_path):
    import subprocess, sys, shutil
    (tmp_path / "data" / "raw").mkdir(parents=True)
    os.symlink(os.path.join(ROOT, "src"), tmp_path / "src")
    shutil.copy(os.path.join(ROOT, "make_data.py"), tmp_path / "make_data.py")
    shutil.copy(os.path.join(ROOT, "config.yaml"), tmp_path / "config.yaml")
    done = subprocess.run([sys.executable, "make_data.py"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and "Missing raw files" in (done.stderr + done.stdout) and "X06.gz" in (done.stderr + done.stdout)
