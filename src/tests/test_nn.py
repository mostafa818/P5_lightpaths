import os
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
