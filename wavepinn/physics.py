"""
Physique de la houle 1D linéaire, eau profonde : vérité terrain η(x, t),
vitesse de groupe et zone de prédictibilité.

Une mer est un dict de composantes {A, omega, k, phi} (+ méta) :
    η(x, t) = Σ Aₙ cos(kₙ x − ωₙ t + φₙ),   kₙ = ωₙ²/g
Le spectre vient de jonswap.spectral_components : même modèle que le jumeau.
"""

import numpy as np

import jonswap

g = jonswap.g

# Nombre de composantes de la vérité : bien plus que les 30 du jumeau, pour
# qu'aucune méthode de reconstruction ne retombe sur ses fréquences exactes
# (« crime inverse »).
N_FREQ_TRUTH = 256


def make_sea_1d(Hs, Tp, gamma, seed, n_freq=N_FREQ_TRUTH):
    """Mer unidirectionnelle (vers +x) JONSWAP, entièrement fixée par la graine."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.5, 0.5, n_freq)
    phi = rng.uniform(0.0, 2 * np.pi, n_freq)
    w, _, k, _, A = jonswap.spectral_components(Hs, Tp, gamma, 0.0, 1.0, jitter, n_dir=1)
    return dict(A=A[:, 0], omega=w, k=k, phi=phi,
                Hs=float(Hs), Tp=float(Tp), gamma=float(gamma), seed=int(seed))


def eta(sea, x, t):
    """Élévation η sur la grille (t, x) : tableau [Nt, Nx] (float64)."""
    x, t = np.atleast_1d(x).astype(float), np.atleast_1d(t).astype(float)
    # Re[ e^{−iωt} · diag(A e^{iφ}) · e^{ikx} ] : un seul produit matriciel
    c = sea['A'] * np.exp(1j * sea['phi'])
    return ((np.exp(-1j * np.outer(t, sea['omega'])) * c)
            @ np.exp(1j * np.outer(sea['k'], x))).real


def group_velocity(omega):
    """c_g = dω/dk = g / (2ω) en eau profonde (= c/2)."""
    # shortcut: eau profonde seulement, à étendre si on passe en profondeur finie
    return g / (2.0 * np.asarray(omega))


def energy_band(Tp, gamma, frac=0.90):
    """
    Bande [ω_lo, ω_hi] contenant `frac` de l'énergie du spectre JONSWAP
    théorique (coupée symétriquement : 5 % / 95 % pour frac = 0.90).
    """
    wp = 2 * np.pi / Tp
    w = np.linspace(0.3 * wp, 6.0 * wp, 20000)
    cdf = np.cumsum(jonswap.jonswap_spectrum(w, 1.0, Tp, gamma))
    cdf /= cdf[-1]
    lo, hi = np.interp([(1 - frac) / 2, (1 + frac) / 2], cdf, w)
    return float(lo), float(hi)


def predictable_window(t0, T, d, w_lo, w_hi):
    """
    Fenêtre de prédictibilité [t_début, t_fin] d'un point situé à la distance
    d (m, en aval) d'une sonde qui a mesuré pendant [t0 − T, t0].

    L'information de chaque composante voyage à c_g ∈ [c_g,min, c_g,max] :
      t_début = t0 − T + d / c_g,min   (les plus lentes ont fini d'arriver)
      t_fin   = t0     + d / c_g,max   (les plus rapides n'ont plus rien apporté)
    Fenêtre vide si t_début > t_fin. d peut être un tableau (une sonde par
    valeur) : on renvoie l'union des fenêtres non vides.
    """
    cg_min, cg_max = group_velocity(w_hi), group_velocity(w_lo)
    d = np.atleast_1d(np.asarray(d, dtype=float))
    start = t0 - T + d / cg_min
    end = t0 + d / cg_max
    ok = start <= end
    if not ok.any():
        return None
    # shortcut: union supposée connexe (sondes rapprochées) ; à revoir pour des sondes très espacées
    return float(start[ok].min()), float(end[ok].max())
