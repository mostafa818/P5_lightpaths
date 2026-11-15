import json, os
import numpy as np, pytest
from nn import features as F, evaluate as E
from nn.model import train_mlp
from nn import forecast as FC

ROOT = FC.ROOT
HOURLY = os.path.join(ROOT, "data", "processed", "hourly.npz")
MODEL = os.path.join(FC.MODEL_DIR, "mlp_seed0.joblib")
needs_data = pytest.mark.skipif(not os.path.exists(HOURLY), reason="run make_data.py first")
needs_model = pytest.mark.skipif(not (os.path.exists(HOURLY) and os.path.exists(MODEL)), reason="run `python -m nn.train` first")


@pytest.fixture(scope="module")
def H():
    return np.load(HOURLY)["hourly"]


@needs_data
def test_feature_shape_and_range(H):
    sc = F.Scaler.fit(H[:432])
    X = F.make_features(H, 200, sc)
    assert X.shape == (132, F.N_FEATURES) == (132, 29)
    assert np.isfinite(X).all()


@needs_data
def test_no_feature_uses_hour_t_or_later(H):
    sc = F.Scaler.fit(H[:432])
    t = 300
    H2 = H.copy()
    H2[t:] = np.random.default_rng(0).random(H2[t:].shape) * 1e4     # corrupt the future
    assert np.array_equal(F.make_features(H, t, sc), F.make_features(H2, t, sc))
    # but history does matter
    H3 = H.copy(); H3[t - 1] += 1000
    assert not np.array_equal(F.make_features(H, t, sc), F.make_features(H3, t, sc))


@needs_data
def test_scaler_uses_training_hours_only_and_inverts(H):
    sc = F.Scaler.fit(H[:432])
    assert sc.mu == pytest.approx(np.log1p(H[:432]).mean())
    assert np.allclose(sc.inverse(sc.transform(H[10])), H[10])


@needs_data
def test_needs_168h_history_and_enough_data(H):
    sc = F.Scaler.fit(H[:432])
    with pytest.raises(ValueError):
        F.make_features(H, 167, sc)
    with pytest.raises(ValueError):
        F.make_features(H, 700, sc)


@needs_data
def test_same_seed_same_result(H):
    sc = F.Scaler.fit(H[:432])
    X, y = F.build_dataset(H, range(168, 220), sc)
    z = sc.transform(y)
    a, _ = train_mlp(3, X, z, X[:200], z[:200], max_epochs=2)
    b, _ = train_mlp(3, X, z, X[:200], z[:200], max_epochs=2)
    c, _ = train_mlp(4, X, z, X[:200], z[:200], max_epochs=2)
    assert np.array_equal(a.predict(X[:50]), b.predict(X[:50]))
    assert not np.array_equal(a.predict(X[:50]), c.predict(X[:50]))


@needs_model
def test_forecast_shape_nonnegative():
    for t in (168, 504, 671):
        f = FC.forecast(t)
        assert f.shape == (132,) and (f >= 0).all() and np.isfinite(f).all()


@needs_model
def test_forecast_ignores_future(H):
    FC._load(0)
    saved = FC._cache["H"]
    try:
        t = 520
        a = FC.forecast(t)
        corrupt = saved.copy(); corrupt[t:] = 9999.0
        FC._cache["H"] = corrupt
        assert np.array_equal(a, FC.forecast(t))
    finally:
        FC._cache["H"] = saved


@needs_model
def test_forecast_rejects_too_early():
    with pytest.raises(ValueError):
        FC.forecast(100)


def test_topk_hit_rate():
    x = np.arange(132, dtype=float)[None, :]
    assert E.topk_hit_rate(x, x, 10) == 1.0
    assert E.topk_hit_rate(x, -x, 10) == 0.0
    m = E.metrics(x, x)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["hit@40"] == 1.0


@needs_data
def test_baselines_use_only_past(H):
    ts = [200, 300]
    assert np.array_equal(E.baseline_last_hour(H, ts)[0], H[199])
    assert np.array_equal(E.baseline_last_week(H, ts)[1], H[132])


@needs_data
def test_early_stopping_restores_best_weights(H):
    sc = F.Scaler.fit(H[:432])
    X, y = F.build_dataset(H, range(168, 300), sc)
    Xv, yv = F.build_dataset(H, range(300, 340), sc)
    z, zv = sc.transform(y), sc.transform(yv)
    m, hist = train_mlp(0, X, z, Xv, zv, max_epochs=30, patience=3)
    best = min(h["val_mse"] for h in hist)
    assert float(np.mean((m.predict(Xv) - zv) ** 2)) == pytest.approx(best)


@needs_model
def test_common_forecast_delegates_to_nn():
    import common
    assert np.array_equal(common.forecast(504), FC.forecast(504, seed=0))


# ---------- Stage 8 additions (Member A) ----------------------------------------------------------------
def test_scaler_json_round_trip(tmp_path):
    sc = F.Scaler(1.25, 0.5)
    sc.to_json(str(tmp_path / "s.json"))
    back = F.Scaler.from_json(str(tmp_path / "s.json"))
    assert (back.mu, back.sd) == (1.25, 0.5)
    x = np.array([0.0, 3.0, 100.0])
    assert np.allclose(back.inverse(back.transform(x)), x)


def test_topk_hits_per_hour_agrees_with_the_mean_hit_rate():
    rng = np.random.default_rng(3)
    true, pred = rng.random((12, 132)), rng.random((12, 132))
    per_hour = E.topk_hits_per_hour(true, pred, 20)
    assert per_hour.shape == (12,) and per_hour.mean() == pytest.approx(E.topk_hit_rate(true, pred, 20))
    assert (per_hour >= 0).all() and (per_hour <= 1).all()


@needs_data
def test_train_script_end_to_end_in_a_temporary_copy(tmp_path, monkeypatch):
    """`python -m nn.train` runs, saves scaler, models, curves and results, and its baselines match a direct computation."""
    import shutil, yaml
    from nn import train
    (tmp_path / "data" / "processed").mkdir(parents=True)
    for f in ("hourly.npz", "meta.json"):
        shutil.copy(os.path.join(ROOT, "data", "processed", f), tmp_path / "data" / "processed" / f)
    with open(os.path.join(ROOT, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cfg["nn"].update({"seeds": [0, 1], "max_epochs": 3, "patience": 2})
    with open(tmp_path / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(train, "ROOT", str(tmp_path))
    train.main()
    out = tmp_path / "outputs"
    assert (out / "nn" / "scaler.json").exists() and (out / "nn" / "mlp_seed1.joblib").exists() and (out / "nn" / "curve_seed0.json").exists()
    with open(out / "nn_eval.json") as fh:
        res = json.load(fh)
    assert set(res["nn_seeds"]) == {"0", "1"} and 0 < res["nn_seeds"]["0"]["epochs"] <= 3
    H = np.load(HOURLY)["hourly"]
    direct = E.metrics(H[504:672], E.baseline_last_hour(H, range(504, 672)))
    assert res["baseline_last_hour"]["mae"] == pytest.approx(direct["mae"])
    assert res["nn_mean"]["mae"] > 0 and set(res["paired_hit_diff_vs_last_hour"]) == {"hit@10", "hit@20", "hit@40"}
