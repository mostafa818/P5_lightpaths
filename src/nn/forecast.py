"""Step 4.7: forecast(t) -> 132 non-negative predicted hourly demands (Mbps) for hour t,
using only data up to hour t-1. Models/scaler are produced by `python -m nn.train`."""
from __future__ import annotations
import os
import joblib
import numpy as np
from nn.features import Scaler, make_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "outputs", "nn")
HOURLY = os.path.join(ROOT, "data", "processed", "hourly.npz")
_cache: dict = {}


def _load(seed: int):
    if "H" not in _cache:
        _cache["H"] = np.load(HOURLY)["hourly"]
        _cache["scaler"] = Scaler.from_json(os.path.join(MODEL_DIR, "scaler.json"))
    if seed not in _cache:
        _cache[seed] = joblib.load(os.path.join(MODEL_DIR, f"mlp_seed{seed}.joblib"))
    return _cache["H"], _cache["scaler"], _cache[seed]


def forecast(t: int, seed: int = 0) -> np.ndarray:
    H, scaler, model = _load(seed)
    z = model.predict(make_features(H, t, scaler))      # reads H[:t] only
    return scaler.inverse(z)                             # expm1 + clip at 0
