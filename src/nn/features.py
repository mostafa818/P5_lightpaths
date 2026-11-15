"""Step 4.1 / 4.2: leakage-safe features and scaling.

Row = (target hour t, pair p). Inputs (29): demand at t-24..t-1, demand at t-168,
sin/cos hour-of-day, sin/cos weekday. Demands are log1p'ed and standardised with
statistics from TRAINING hours only. make_features reads only rows < t of the data.
"""
from __future__ import annotations
import json
import numpy as np

N_PAIRS, N_LAGS, WEEK = 132, 24, 168
N_FEATURES = N_LAGS + 1 + 4
START_WEEKDAY = 5            # hour 0 = 2004-05-01 00:00 = Saturday (Mon=0), see meta.json


class Scaler:
    def __init__(self, mu: float = 0.0, sd: float = 1.0):
        self.mu, self.sd = float(mu), float(sd)

    @classmethod
    def fit(cls, H_train: np.ndarray) -> "Scaler":
        z = np.log1p(H_train)
        return cls(z.mean(), z.std())

    def transform(self, x):
        return (np.log1p(x) - self.mu) / self.sd

    def inverse(self, z):
        return np.clip(np.expm1(np.asarray(z) * self.sd + self.mu), 0.0, None)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump({"mu": self.mu, "sd": self.sd}, fh)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["mu"], d["sd"])


def calendar(t: int) -> np.ndarray:
    hod = t % 24
    wd = (START_WEEKDAY + t // 24) % 7
    return np.array([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                     np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7)])


def make_features(H: np.ndarray, t: int, scaler: Scaler) -> np.ndarray:
    """(132, 29) feature rows for target hour t. Uses ONLY H[:t] (no future)."""
    if t < WEEK:
        raise ValueError(f"need {WEEK} hours of history, got t={t}")
    hist = H[:t]                                   # everything >= t is invisible from here on
    if hist.shape[0] != t:
        raise ValueError(f"data has only {H.shape[0]} hours, cannot forecast hour {t}")
    lags = scaler.transform(hist[-N_LAGS:]).T      # (132, 24): t-24 ... t-1
    week = scaler.transform(hist[t - WEEK])[:, None]   # (132, 1): same hour last week
    cal = np.tile(calendar(t), (H.shape[1], 1))
    return np.hstack([lags, week, cal])


def build_dataset(H: np.ndarray, ts, scaler: Scaler):
    """Stack rows for target hours ts: X (len(ts)*132, 29), y_raw (len(ts)*132,) in Mbps."""
    X = np.vstack([make_features(H, int(t), scaler) for t in ts])
    y = np.concatenate([H[int(t)] for t in ts])
    return X, y
