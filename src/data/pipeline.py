"""Data pipeline (Member A): raw Zhang Abilene weekly files -> hourly (n_hours, 132) Mbps.

Format (data/raw/readme.txt): each X*.gz has 2016 lines (12x24x7 5-min TMs); each line
holds 144 x 5 values = for every OD pair [real, simpleGravity, simpleTomogravity,
generalGravity, generalTomogravity]. Unit: 100 bytes / 5 min. Row = sender, col = receiver,
router order alphabetical (verified in Step 1.9, see make_data.py output).
"""
from __future__ import annotations
import gzip
import numpy as np

N_NODES = 12
SLOTS_PER_HOUR = 12
SLOTS_PER_WEEK = 2016
MBPS_FACTOR = 100 * 8 / (300 * 1e6)   # (100 bytes / 5 min) -> Mbps


def load_week_real(path: str) -> np.ndarray:
    """Return the REAL measured block only, shape (2016, 144), raw units."""
    with gzip.open(path, "rt") as fh:
        raw = np.loadtxt(fh)
    if raw.shape != (SLOTS_PER_WEEK, N_NODES * N_NODES * 5):
        raise ValueError(f"{path}: unexpected shape {raw.shape}")
    blocks = raw.reshape(SLOTS_PER_WEEK, N_NODES * N_NODES, 5)
    return blocks[:, :, 0]


def to_mbps(x: np.ndarray) -> np.ndarray:
    return x * MBPS_FACTOR


def to_hourly(x5: np.ndarray) -> np.ndarray:
    """(n_slots, 144) 5-min rates -> (n_slots/12, 144) hourly MEAN rates."""
    n = x5.shape[0]
    if n % SLOTS_PER_HOUR:
        raise ValueError("number of 5-min slots must be a multiple of 12")
    return x5.reshape(n // SLOTS_PER_HOUR, SLOTS_PER_HOUR, x5.shape[1]).mean(axis=1)


def pair_list(node_order: list[str]) -> list[tuple[str, str]]:
    """132 ordered (src, dst) pairs, src-major, diagonal removed."""
    return [(s, d) for s in node_order for d in node_order if s != d]


def drop_diagonal(h144: np.ndarray) -> np.ndarray:
    keep = [i * N_NODES + j for i in range(N_NODES) for j in range(N_NODES) if i != j]
    return h144[:, keep]


def sanity_check(h: np.ndarray) -> dict:
    return {
        "n_nan": int(np.isnan(h).sum()),
        "n_negative": int((h < 0).sum()),
        "n_allzero_rows": int((h == 0).all(axis=1).sum()),      # not sum()==0: positives and negatives could cancel
        "n_allzero_pairs": int((h == 0).all(axis=0).sum()),
    }
