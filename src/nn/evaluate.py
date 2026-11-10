"""Step 4.3 / 4.5: baselines and metrics (original Mbps scale)."""
from __future__ import annotations
import numpy as np

KS = (10, 20, 40)


def topk_hit_rate(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    """true, pred: (T, 132). Mean over hours of |top-k(true) & top-k(pred)| / k."""
    hits = []
    for tr, pr in zip(true, pred):
        a = set(np.argsort(-tr, kind="stable")[:k])
        b = set(np.argsort(-pr, kind="stable")[:k])
        hits.append(len(a & b) / k)
    return float(np.mean(hits))


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    out = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
    for k in KS:
        out[f"hit@{k}"] = topk_hit_rate(true, pred, k)
    return out


def baseline_last_hour(H: np.ndarray, ts) -> np.ndarray:
    return np.stack([H[t - 1] for t in ts])


def baseline_last_week(H: np.ndarray, ts) -> np.ndarray:
    return np.stack([H[t - 168] for t in ts])
