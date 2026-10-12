"""QCA surrogate v3: bounded output via tanh."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qca import hamiltonian, ground_state, polarization


def exact_response(t, V, d):
    H, B = hamiltonian(t=t, V_diag=V, V_adj=0.2, driver=d)
    _, psi = ground_state(H)
    return polarization(psi, B)


def feats(X):
    t, V, d = X[:, 0], X[:, 1], X[:, 2]
    eps = 1e-6
    r = d / (t + eps)
    return np.stack([np.tanh(r), np.tanh(2 * r),
                     np.sqrt(d*d + t*t), V / (V + t + eps),
                     t, d, V], axis=1)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack([rng.uniform(0.01, 0.5, n),
                  rng.uniform(0.5, 2.0, n),
                  rng.uniform(-3.0, 3.0, n)], axis=1)
    y = np.array([exact_response(*x) for x in X])
    return X, y


def fit_torch(X, y, epochs=600):
    try:
        import torch, torch.nn as nn
    except ImportError:
        return None
    F = torch.tensor(feats(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    mu = F.mean(0, keepdim=True); sd = F.std(0, keepdim=True) + 1e-6
    Fn = (F - mu) / sd
    net = nn.Sequential(
        nn.Linear(7, 48), nn.Tanh(),
        nn.Linear(48, 48), nn.Tanh(),
        nn.Linear(48, 1), nn.Tanh(),     # <-- bounded output
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Fn), yt)
        loss.backward(); opt.step()
        if (ep + 1) % 200 == 0:
            print(f"  epoch {ep+1:>4}  loss = {loss.item():.6f}")
    return net, mu, sd


if __name__ == "__main__":
    print("=== Surrogate v3 ===\n")
    X, y = generate(1500, seed=0)
    out = fit_torch(X, y, epochs=600)
    if out is None:
        print("torch unavailable"); sys.exit(0)
    net, mu, sd = out
    import torch
    with torch.no_grad():
        pred = net((torch.tensor(feats(X), dtype=torch.float32) - mu) / sd).squeeze().numpy()
    print(f"\n  train RMSE = {np.sqrt(np.mean((pred - y) ** 2)):.5f}")

    Xv, yv = generate(300, seed=99)
    with torch.no_grad():
        pv = net((torch.tensor(feats(Xv), dtype=torch.float32) - mu) / sd).squeeze().numpy()
    print(f"  held-out RMSE = {np.sqrt(np.mean((pv - yv) ** 2)):.5f}")
    print(f"  max |P_pred| = {np.abs(pv).max():.4f}  (must be <= 1)")
