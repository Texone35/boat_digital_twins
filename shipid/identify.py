"""
Identification de (ω_n, ζ, b_q) dans l'équation de roulis à 1 DDL
    φ̈ + 2ζω_n φ̇ + ω_n²(φ − φ_w) + b_q |φ̇| φ̇ = 0
à partir du roulis mesuré φ(t) et de la pente de houle φ_w(t).

Trois méthodes, même sortie dict(omega_n, zeta, b_q, seconds) :
  equation_error : dérivées numériques (Savitzky-Golay) + régression linéaire.
                   Immédiat, mais dériver deux fois un signal bruité biaise.
  output_error   : on simule l'équation et on ajuste ses paramètres pour
                   reproduire φ mesuré (référence classique de l'identification).
  pinn           : réseau φ_θ(t) ajusté aux mesures sous contrainte du résidu de
                   l'équation (dérivées exactes par autograd, pas de dérivée
                   numérique des mesures) ; paramètres physiques entraînables.
"""

import time

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


def _result(omega_n, zeta, b_q, start, **extra):
    return dict(omega_n=float(omega_n), zeta=float(zeta), b_q=float(b_q),
                seconds=round(time.time() - start, 2), **extra)


def equation_error(t, roll, phi_w, window_s=1.5):
    start = time.time()
    dt = t[1] - t[0]
    n = int(window_s / dt) | 1                          # fenêtre impaire
    d0, d1, d2 = (savgol_filter(roll, n, 3, deriv=k, delta=dt) for k in (0, 1, 2))
    w = savgol_filter(phi_w, n, 3)
    # φ̈ = −c1 φ̇ − c2 (φ − φ_w) − c3 |φ̇| φ̇
    X = -np.column_stack([d1, d0 - w, np.abs(d1) * d1])
    c1, c2, c3 = np.linalg.lstsq(X, d2, rcond=None)[0]
    omega = np.sqrt(max(c2, 1e-12))
    return _result(omega, c1 / (2 * omega), c3, start)


def simulate(t, phi_w, omega_n, zeta, b_q, phi0=0.0, dphi0=0.0):
    """Roulis de l'équation à 1 DDL sur la grille t (RK4, φ_w linéaire par morceaux)."""
    dt = t[1] - t[0]
    w_mid = 0.5 * (phi_w[:-1] + phi_w[1:])
    a, b = 2 * zeta * omega_n, omega_n**2

    def f(p, v, w):
        return v, -a * v - b * (p - w) - b_q * abs(v) * v

    out = np.empty(len(t))
    p, v = phi0, dphi0
    out[0] = p
    for i in range(len(t) - 1):
        w0, wm, w1 = phi_w[i], w_mid[i], phi_w[i + 1]
        k1 = f(p, v, w0)
        k2 = f(p + 0.5*dt*k1[0], v + 0.5*dt*k1[1], wm)
        k3 = f(p + 0.5*dt*k2[0], v + 0.5*dt*k2[1], wm)
        k4 = f(p + dt*k3[0], v + dt*k3[1], w1)
        p += dt / 6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        v += dt / 6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        out[i + 1] = p
    return out


def output_error(t, roll, phi_w, init):
    """init : dict(omega_n, zeta, b_q) de départ (en pratique : equation_error)."""
    start = time.time()
    x0 = [init['omega_n'], max(init['zeta'], 1e-3), max(init['b_q'], 0.0), roll[0], 0.0]
    res = least_squares(lambda x: simulate(t, phi_w, *x) - roll, x0,
                        bounds=([1e-3, 0, 0, -np.inf, -np.inf], np.inf), x_scale='jac')
    return _result(*res.x[:3], start, phi0=float(res.x[3]), dphi0=float(res.x[4]))


def pinn(t, roll, phi_w, init, iters=3000, lr=3e-3, n_ff=64, hidden=64, w_phys=1.0, seed=0,
         log_every=0):
    """
    PINN d'identification. init : modèle nominal dict(omega_n, zeta, b_q) — point
    de départ des paramètres physiques (le navire tel que dessiné).
    Unités réduites : t̃ = t / T0 (T0 = 2π/ω_n nominal), φ̃ = φ / std(φ) ;
    résidu  φ̃'' + 2ζΩ φ̃' + Ω²(φ̃ − φ̃_w) + b_q·s·|φ̃'|φ̃' = 0,  Ω = ω_n·T0.
    """
    import torch
    from torch import nn

    start = time.time()
    torch.manual_seed(seed)
    T0 = 2 * np.pi / init['omega_n']
    s = float(np.std(roll))
    col = lambda a: torch.tensor(np.asarray(a, np.float32).reshape(-1, 1))
    tt = col((t - t[0]) / T0).requires_grad_()
    y, yw = col(roll / s), col(phi_w / s)

    B = torch.rand(1, n_ff) * 3.0                       # 0-3 cycles par période propre nominale
    net = nn.Sequential(nn.Linear(2 * n_ff, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                        nn.Tanh(), nn.Linear(hidden, 1))

    def phi(x):
        arg = 2 * np.pi * x @ B
        return net(torch.cat([torch.sin(arg), torch.cos(arg)], 1))

    inv_softplus = lambda v: float(np.log(np.expm1(max(v, 1e-4))))
    log_Om = nn.Parameter(torch.tensor(float(np.log(2 * np.pi))))
    raw_zeta = nn.Parameter(torch.tensor(inv_softplus(init['zeta'])))
    raw_bq = nn.Parameter(torch.tensor(inv_softplus(init['b_q'] + 1e-2)))
    sp = nn.functional.softplus
    opt = torch.optim.Adam([*net.parameters(), log_Om, raw_zeta, raw_bq], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    scale = (2 * np.pi) ** 2                            # Ω² nominal : résidu relatif

    for it in range(iters):
        p = phi(tt)
        d1 = torch.autograd.grad(p, tt, torch.ones_like(p), create_graph=True)[0]
        d2 = torch.autograd.grad(d1, tt, torch.ones_like(d1), create_graph=True)[0]
        Om, zeta, bq = torch.exp(log_Om), sp(raw_zeta), sp(raw_bq)
        r = (d2 + 2 * zeta * Om * d1 + Om**2 * (p - yw) + bq * s * d1.abs() * d1) / scale
        l_data, l_phys = ((p - y) ** 2).mean(), (r ** 2).mean()
        loss = l_data + w_phys * l_phys
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and it % log_every == 0:
            print(f'it {it:5d} data {l_data.item():.2e} phys {l_phys.item():.2e} '
                  f'ω_n {Om.item() / T0:.4f} ζ {zeta.item():.4f} b_q {bq.item():.3f}', flush=True)
    return _result(torch.exp(log_Om).item() / T0, sp(raw_zeta).item(), sp(raw_bq).item(), start,
                   fit=p.detach().numpy().ravel() * s)
