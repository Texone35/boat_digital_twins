"""
Opérateur neuronal d'identification du roulis : un FNO 1D (Li et al. 2021)
apprend UNE FOIS, sur des milliers de navires et de mers simulés,
l'application  (roulis φ(t), pente de houle φ_w(t))  →  (ω_n, ζ, b_q).
Ensuite chaque identification est une simple évaluation du réseau
(millisecondes), là où le PINN se réentraîne pour chaque enregistrement.

Données d'entraînement : équation de roulis à 1 DDL (cf. identify.simulate),
intégrée en RK4 vectorisé sur tout le lot ; houle JONSWAP aléatoire ; bruit de
mesure 0-1°. 300 s à 1 Hz (300 points) par exemple. Plage d'entraînement :
RANGES — hors de cette plage, l'opérateur n'a rien appris (contrairement au PINN
ou à l'erreur de sortie, qui s'ajustent à chaque enregistrement).
"""

import copy
import time

import numpy as np
import torch
from torch import nn

import jonswap

FS = 1.0                 # Hz, entrée de l'opérateur (roulis < 0.15 Hz, houle utile < 0.3 Hz)
DURATION = 300.0         # s
N_T = int(FS * DURATION)
SCALE = 10.0             # rad → unités du réseau (échelle fixe : l'amplitude porte b_q)
RANGES = dict(omega_n=(0.25, 0.8), zeta=(0.02, 0.15), b_q=(0.0, 2.5))


# ── Données synthétiques ──
def synthetic_batch(n, rng, dt=0.1, n_comp=48):
    """
    n enregistrements (roll [n, N_T], phi_w [n, N_T]) et leurs paramètres
    [n, 3] = (ω_n, ζ, b_q). Simulation à dt, puis moyenne par blocs → FS.
    """
    t = np.arange(0.0, DURATION, dt)
    Tp = rng.uniform(6.0, 18.0, (n, 1))
    w = (2 * np.pi / Tp) * rng.uniform(0.6, 3.0, (n, n_comp))
    a = np.sqrt(jonswap.jonswap_spectrum(w, 1.0, Tp, 3.3))
    ph = rng.uniform(0, 2 * np.pi, (n, n_comp))
    phi_w = np.zeros((n, len(t)))
    for k in range(n_comp):
        phi_w += a[:, k:k + 1] * np.cos(w[:, k:k + 1] * t + ph[:, k:k + 1])
    amp = np.exp(rng.uniform(np.log(np.radians(0.3)), np.log(np.radians(4.0)), (n, 1)))
    phi_w *= amp / phi_w.std(axis=1, keepdims=True)

    lo, hi = np.array([RANGES[k] for k in RANGES]).T
    params = lo + (hi - lo) * rng.uniform(size=(n, 3))
    params[:, 0] = np.exp(np.log(lo[0]) + np.log(hi[0] / lo[0]) * rng.uniform(size=n))
    om, ze, bq = params.T
    a1, a2 = 2 * ze * om, om ** 2

    def f(p, v, u):
        return v, -a1 * v - a2 * (p - u) - bq * np.abs(v) * v

    p, v = np.zeros(n), np.zeros(n)
    roll = np.empty_like(phi_w)
    for i in range(len(t)):                       # RK4, φ_w tenue sur le pas
        roll[:, i] = p
        u = phi_w[:, i]
        k1 = f(p, v, u)
        k2 = f(p + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], u)
        k3 = f(p + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], u)
        k4 = f(p + dt * k3[0], v + dt * k3[1], u)
        p = p + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    roll += rng.uniform(0, np.radians(1.0), (n, 1)) * rng.standard_normal(roll.shape)
    return decimate(roll, dt), decimate(phi_w, dt), params


def decimate(x, dt):
    """Moyenne par blocs de dt vers FS (anti-repliement simple)."""
    k = int(round(1.0 / (FS * dt)))
    x = np.asarray(x)[..., :N_T * k]
    return x.reshape(*x.shape[:-1], N_T, k).mean(-1)


def _to_unit(params):
    lo, hi = np.array([RANGES[k] for k in RANGES]).T
    u = (params - lo) / (hi - lo)
    u[:, 0] = np.log(params[:, 0] / lo[0]) / np.log(hi[0] / lo[0])
    return u


def _from_unit(u):
    lo, hi = np.array([RANGES[k] for k in RANGES]).T
    p = lo + (hi - lo) * u
    p[:, 0] = lo[0] * (hi[0] / lo[0]) ** u[:, 0]
    return p


# ── Réseau ──
class SpectralConv1d(nn.Module):
    """Convolution dans l'espace de Fourier : produit par des poids complexes
    sur les `modes` premières fréquences (couche de base du FNO)."""

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, x):
        X = torch.fft.rfft(x)
        out = torch.zeros_like(X)
        out[..., :self.modes] = torch.einsum('bix,iox->box', X[..., :self.modes], self.w)
        return torch.fft.irfft(out, n=x.shape[-1])


class RollFNO(nn.Module):
    def __init__(self, width=24, modes=48, layers=3):
        super().__init__()
        self.lift = nn.Conv1d(2, width, 1)
        self.spectral = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, x):                       # x : (B, 2, N)
        h = self.lift(x)
        for s, c in zip(self.spectral, self.local):
            h = nn.functional.gelu(s(h) + c(h))
        return self.head(torch.cat([h.mean(-1), h.std(-1)], 1))


def _inputs(roll, phi_w):
    return torch.tensor(np.stack([roll, phi_w], 1) * SCALE, dtype=torch.float32)


def train(n_train=8000, n_val=1000, epochs=30, batch=64, lr=2e-3, seed=0, log=print):
    """Génère les données et entraîne l'opérateur. Retourne (modèle, infos)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    start = time.time()
    roll, phi_w, params = synthetic_batch(n_train + n_val, rng)
    t_data = time.time() - start
    X, Y = _inputs(roll, phi_w), torch.tensor(_to_unit(params), dtype=torch.float32)
    Xv, Yv, X, Y = X[n_train:], Y[n_train:], X[:n_train], Y[:n_train]

    model = RollFNO()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (n_train // batch))
    best = (np.inf, 0, None)                    # (perte val, époque, poids) : meilleur état gardé
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(n_train // batch):
            idx = perm[i * batch:(i + 1) * batch]
            loss = ((model(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            val = ((model(Xv) - Yv) ** 2).mean().item()
        if val < best[0]:
            best = (val, ep, copy.deepcopy(model.state_dict()))
        if log and (ep % 5 == 0 or ep == epochs - 1):
            log(f'époque {ep:3d}  perte val {val:.4f}  ({time.time() - start:.0f} s)')
    model.load_state_dict(best[2])
    return model, dict(data_s=round(t_data, 1), train_s=round(time.time() - start - t_data, 1),
                       n_train=n_train, epochs=epochs, best_epoch=best[1],
                       val_loss=round(best[0], 4))


@torch.no_grad()
def predict(model, roll, phi_w):
    """roll, phi_w : [n, N_T] ou [N_T] à FS → dict de tableaux omega_n, zeta, b_q."""
    roll, phi_w = np.atleast_2d(roll), np.atleast_2d(phi_w)
    model.eval()
    p = _from_unit(model(_inputs(roll, phi_w)).numpy().astype(float))
    return dict(omega_n=p[:, 0], zeta=p[:, 1], b_q=p[:, 2])


def save(model, path, info):
    torch.save(dict(state=model.state_dict(), info=info), path)


def load(path):
    d = torch.load(path, weights_only=False)
    model = RollFNO()
    model.load_state_dict(d['state'])
    return model, d['info']
