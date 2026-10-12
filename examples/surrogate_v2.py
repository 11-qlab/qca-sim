"""QCA surrogate with physics-motivated features.

The old (t, V, driver) -> MLP failed because the sharp transition
near driver/t ~ 1 is invisible in raw coordinates.  Feeding
dimensionless ratios fixes it.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qca import hamiltonian, ground_state, polarization


def exact_response(t, V_diag, driver):
    H, B = hamiltonian(t=t, V_diag=V_diag, V_adj=0.2, driver=driver)
    _, psi = ground_state(H)
    return polarization(psi, B)


def phys_features(X):
    """Map raw (t, V, driver) to dimensionless physics features."""
    t, V, d = X[:, 0], X[:, 1], X[:, 2]
    eps = 1e-6
    ratio = d / (t + eps)
    return np.stack([
        np.tanh(ratio),                       # saturated driver
        ratio / (1 + np.abs(ratio)),          # bounded ratio
        np.sqrt(d * d + t * t),               # gap
        V / (V + t + eps),                    # V/t competition
        t,
        d,
        V,
    ], axis=1)


def generate(n, seed):
    rng = np.random.default_rng(seed)
    X = np.stack([
        rng.uniform(0.01, 0.5, n),
        rng.uniform(0.5, 2.0, n),
        rng.uniform(-3.0, 3.0, n),
    ], axis=1)
    y = np.array([exact_response(*x) for x in X])
    return X, y


def fit_numpy(X, y):
    F = phys_features(X)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return w


def fit_torch(X, y, epochs=400):
    try:
        import torch, torch.nn as nn
    except ImportError:
        return None
    Xt = torch.tensor(phys_features(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    mu = Xt.mean(0, keepdim=True); sd = Xt.std(0, keepdim=True) + 1e-6
    Xn = (Xt - mu) / sd
    model = nn.Sequential(
        nn.Linear(7, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(model(Xn), yt)
        loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:>4}  loss = {loss.item():.6f}")
    return model, mu, sd


if __name__ == "__main__":
    print("=== Surrogate v2 (physics features) ===\n")
    print("generating training data...")
    X, y = generate(1500, seed=0)

    model = fit_torch(X, y, epochs=400)
    if model is None:
        print("torch unavailable -- numpy polynomial fallback")
        w = fit_numpy(X, y)
        pred = phys_features(X) @ w
        print(f"  train RMSE = {np.sqrt(np.mean((pred - y) ** 2)):.5f}")
        Xv, yv = generate(300, seed=99)
        pv = phys_features(Xv) @ w
        print(f"  held-out RMSE = {np.sqrt(np.mean((pv - yv) ** 2)):.5f}")
        np.save("qca_surrogate_v2.npy", w)
    else:
        m, mu, sd = model
        import torch
        with torch.no_grad():
            p = m((torch.tensor(phys_features(X), dtype=torch.float32) - mu) / sd).squeeze().numpy()
        print(f"\n  train RMSE = {np.sqrt(np.mean((p - y) ** 2)):.5f}")
        Xv, yv = generate(300, seed=99)
        with torch.no_grad():
            pv = m((torch.tensor(phys_features(Xv), dtype=torch.float32) - mu) / sd).squeeze().numpy()
        print(f"  held-out RMSE = {np.sqrt(np.mean((pv - yv) ** 2)):.5f}")

    # Demo: rapid parameter sweep
    print("\n=== Rapid sweep using the surrogate ===")
    print(f"{'t':>6} {'V':>6} {'driver':>8} {'P_pred':>8} {'P_exact':>8}")
    rng = np.random.default_rng(7)
    for _ in range(8):
        t = rng.uniform(0.01, 0.5)
        V = rng.uniform(0.5, 2.0)
        d = rng.uniform(-3, 3)
        if model is None:
            p = phys_features(np.array([[t, V, d]])) @ w
        else:
            with torch.no_grad():
                p = m((torch.tensor(phys_features(np.array([[t, V, d]])), dtype=torch.float32) - mu) / sd).item()
        pe = exact_response(t, V, d)
        print(f"{t:>6.3f} {V:>6.3f} {d:>+8.3f} {float(p):>+8.4f} {pe:>+8.4f}")
