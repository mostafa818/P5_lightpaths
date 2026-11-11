"""Step 4.4: MLP (2x64 ReLU, Adam, batch 256, MSE on log scale) with early stopping on the
chronological validation set (not sklearn's random internal split)."""
from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor


def train_mlp(seed, Xtr, ytr, Xva, yva, hidden=(64, 64), lr=1e-3, batch_size=256,
              max_epochs=100, patience=10):
    m = MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="relu", solver="adam",
                     learning_rate_init=lr, batch_size=batch_size, shuffle=True,
                     random_state=seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        m.partial_fit(Xtr, ytr)
        val = float(np.mean((m.predict(Xva) - yva) ** 2))
        hist.append({"epoch": ep + 1, "train_loss": float(m.loss_), "val_mse": val})
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = ([c.copy() for c in m.coefs_], [b.copy() for b in m.intercepts_])
        else:
            bad += 1
            if bad >= patience:
                break
    m.coefs_, m.intercepts_ = best_state
    return m, hist
