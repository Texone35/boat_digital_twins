"""
Évaluation d'un prédicteur de houle contre la vérité (seule étape qui y accède).

Un prédicteur est une fonction  forecaster(t_win, x_probes, eta_win) -> f,
avec f(x, t) -> η̂ [Nt, Nx]. (baseline_lsq.forecaster).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from waveforecast import physics

USEFUL_THRESHOLD = 0.20   # horizon utile : RMS/Hs dépasse 20 % (cf. document)


def forecast_errors(sea, sens, forecaster, T, t0s, taus, x_target=0.0):
    """
    Prévisions glissantes au point x_target : pour chaque t0, ajustement sur
    les mesures de [t0 − T, t0[, prédiction en t0 + τ.
    Retourne (pred, truth), chacun [len(t0s), len(taus)], normalisés par Hs.
    """
    preds, truths = [], []
    for t0 in t0s:
        m = (sens['t'] >= t0 - T) & (sens['t'] < t0)
        f = forecaster(sens['t'][m], sens['x'], sens['eta'][:, m])
        preds.append(f([x_target], t0 + taus)[:, 0])
        truths.append(physics.eta(sea, [x_target], t0 + taus)[:, 0])
    return np.array(preds) / sea['Hs'], np.array(truths) / sea['Hs']


def horizon_metrics(pred, truth):
    """RMS/Hs et corrélation prédit/vrai pour chaque τ (lignes = prévisions)."""
    rms = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    p, q = pred - pred.mean(0), truth - truth.mean(0)
    corr = (p * q).sum(0) / np.sqrt((p**2).sum(0) * (q**2).sum(0))
    return rms, corr


def useful_horizon(taus, rms, threshold=USEFUL_THRESHOLD):
    """Premier τ ≥ 0 où l'erreur dépasse le seuil (dernier τ sinon)."""
    over = (taus >= 0) & (rms > threshold)
    return float(taus[over][0]) if over.any() else float(taus[-1])


def zone_bounds(x, x_probes, t0, T, w_lo, w_hi):
    """Bornes temporelles de la zone de prédictibilité pour chaque x (nan si aucune)."""
    x_probes = np.asarray(x_probes)
    out = np.full((2, len(x)), np.nan)
    for i, xi in enumerate(x):
        d = xi - x_probes[x_probes <= xi]
        win = physics.predictable_window(t0, T, d, w_lo, w_hi) if len(d) else None
        if win:
            out[:, i] = win
    return out


def error_map(sea, f, x, t):
    """|erreur| locale (RMS glissante sur Tp) / Hs sur la grille (t, x)."""
    err2 = (f(x, t) - physics.eta(sea, x, t)) ** 2
    n = max(1, int(round(sea['Tp'] / (t[1] - t[0]))))
    return np.sqrt(uniform_filter1d(err2, n, axis=0)) / sea['Hs']


def plot_error_map(ax, emap, x, t, t0, T, x_probes, w_lo, w_hi):
    """Carte (x, t) de l'erreur avec la zone théorique superposée."""
    im = ax.pcolormesh(x, t - t0, emap, cmap='magma', vmin=0, vmax=0.4, shading='auto')
    lo, hi = zone_bounds(x, x_probes, t0, T, w_lo, w_hi)
    ax.plot(x, lo - t0, 'c--', lw=1.5, label='zone : pente $c_{g,min}$')
    ax.plot(x, hi - t0, 'c-', lw=1.5, label='zone : pente $c_{g,max}$')
    ax.vlines(x_probes, -T, 0, colors='w', lw=1, label='sondes (mesures)')
    ax.axhline(0, color='w', ls=':', lw=1)
    ax.axvline(0, color='lime', ls=':', lw=1, label='navire')
    ax.set(xlabel='x (m)', ylabel='t − t₀ (s)')
    ax.legend(loc='lower right', fontsize=8)
    return im
