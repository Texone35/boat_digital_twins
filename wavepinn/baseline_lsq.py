"""
Baseline : reconstruction linéaire par moindres carrés régularisés.

    η̂(x, t) = Σₘ aₘ cos(kₘx − ωₘt) + bₘ sin(kₘx − ωₘt),   kₘ = ωₘ²/g

Les ωₘ sont espacées de 2π/T (résolution fréquentielle de la fenêtre de
mesure T) sur la bande [w_lo, w_hi] : grille propre au modèle, sans lien
avec les fréquences de la vérité. Problème mal conditionné → Tikhonov :
    min ‖Ψc − η‖² + α·P‖c‖²   (P = nombre de mesures)
résolu par lstsq sur le système augmenté.
"""

import numpy as np

from wavepinn.physics import energy_band, eta, g

# Bande de reconstruction : 99 % de l'énergie. Avec 90 % (la bande de la zone
# de prédictibilité), les 10 % ignorés laissent un plancher d'erreur ≈ 8-10 % Hs.
RECON_FRAC = 0.99


def fit(t, x, eta_mes, w_lo, w_hi, alpha=1e-3):
    """
    t [Nt], x [Np], eta_mes [Np, Nt] : mesures des sondes sur une fenêtre.
    Retourne une mer {A, omega, k, phi}, utilisable par physics.eta.
    """
    T = t[-1] - t[0]
    w = np.arange(w_lo, w_hi, 2 * np.pi / T)
    k = w**2 / g
    xx, tt = np.meshgrid(x, t, indexing='ij')                    # [Np, Nt]
    ph = np.outer(xx.ravel(), k) - np.outer(tt.ravel(), w)       # [P, M]
    psi = np.hstack([np.cos(ph), np.sin(ph)])
    P, n = psi.shape
    lhs = np.vstack([psi, np.sqrt(alpha * P) * np.eye(n)])
    rhs = np.concatenate([np.asarray(eta_mes, float).ravel(), np.zeros(n)])
    c = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    # a cos θ + b sin θ = Re[(a − ib) e^{iθ}] = A cos(θ + φ)
    z = c[:len(w)] - 1j * c[len(w):]
    return dict(A=np.abs(z), omega=w, k=k, phi=np.angle(z))


def forecaster(Tp, gamma, alpha=1e-3, frac=RECON_FRAC):
    """Prédicteur au format evaluate : (t, x, η) → f(x, t) = η̂ [Nt, Nx]."""
    w_lo, w_hi = energy_band(Tp, gamma, frac)

    def run(t, x, eta_mes):
        model = fit(t, x, eta_mes, w_lo, w_hi, alpha)
        return lambda xs, ts: eta(model, xs, ts)
    return run
