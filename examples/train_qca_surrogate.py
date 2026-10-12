"""Train a small MLP surrogate for the QCA cell response.

Generates training data from exact diagonalization, then fits
a 3 -> 32 -> 32 -> 1 network that predicts polarization P from
(t, V_diag, driver).  Useful for scanning large parameter
spaces without re-diagonalizing.

Falls back to a numpy least-squares fit if torch is absent.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def exact_response(t, V_diag, driver):
    H, B = hamiltonian(t=t, V_diag=V_diag, V_adj=0.2, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def generate(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack([
        rng.uniform(0.01, 0.5, n),      # t
        rng.uniform(0.5, 2.0, n),       # V_diag
        rng.uniform(-3.0, 3.0, n),      # driver
    ], axis=1)
    y = np.array([exact_response(*x) for x in X])
    return X, y


def fit_torch(X, y, epochs=200):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Normalize inputs
    mu = Xt.mean(0, keepdim=True)
    sd = Xt.std(0, keepdim=True) + 1e-6
    Xn = (Xt - mu) / sd

    model = nn.Sequential(
        nn.Linear(3, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xn)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>4}  loss = {loss.item():.6f}")

    return model, mu, sd


def fit_numpy(X, y):
    """Cheap fallback: polynomial features + linear regression."""
    def feats(X):
        t, V, d = X[:, 0], X[:, 1], X[:, 2]
        return np.stack([
            np.ones_like(t), t, V, d,
            t * t, V * V, d * d,
            t * d, V * d, t * V,
            np.tanh(d), np.tanh(2 * d),
        ], axis=1)
    F = feats(X)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return w, feats


if __name__ == "__main__":
    print("=== QCA surrogate: training ===\n")
    print("generating exact data...")
    X, y = generate(n=1500)
    print(f"  got {len(X)} samples\n")

    model = fit_torch(X, y, epochs=200)

    if model is None:
        print("torch not installed -- using numpy polynomial fallback")
        w, feats = fit_numpy(X, y)
        pred = feats(X) @ w
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        print(f"  polynomial RMSE = {rmse:.5f}")
        # Save coefficients
        np.save("qca_surrogate_coefs.npy", w)
        print("  saved qca_surrogate_coefs.npy")
    else:
        m, mu, sd = model
        import torch
        with torch.no_grad():
            pred = m((torch.tensor(X, dtype=torch.float32) - mu) / sd).squeeze().numpy()
        rmse = np.sqrt(np.mean((pred - y) ** 2))
        print(f"\n  MLP RMSE = {rmse:.5f}")

        # Held-out validation
        Xv, yv = generate(n=300, seed=99)
        with torch.no_grad():
            pv = m((torch.tensor(Xv, dtype=torch.float32) - mu) / sd).squeeze().numpy()
        print(f"  held-out RMSE = {np.sqrt(np.mean((pv - yv) ** 2)):.5f}")
