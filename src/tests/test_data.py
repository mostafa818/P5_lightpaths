import os, json
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
    m = json.load(open(os.path.join(ROOT, "data", "processed", "meta.json")))
    assert len(m["pairs"]) == 132 and len(m["node_order"]) == 12
    assert m["split"]["train"] == [0, 503] and m["split"]["test"] == [504, 671]
    assert m["split"]["val"] == [432, 503] and m["start_weekday_mon0"] == 5   # 2004-05-01 = Saturday


@needs_processed
@needs_raw
def test_hourly_matches_manual_average():
    H = np.load(HOURLY)["hourly"]
    x = P.to_mbps(P.load_week_real(RAW_X06))          # (2016, 144) Mbps
    manual = x[12:24, 1 * 12 + 2].mean()               # hour 1, pair (router 1 -> router 2)
    m = json.load(open(os.path.join(ROOT, "data", "processed", "meta.json")))
    k = m["pairs"].index([m["node_order"][1], m["node_order"][2]])
    assert H[1, k] == pytest.approx(manual)
