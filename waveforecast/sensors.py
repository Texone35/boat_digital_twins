"""
Capteurs virtuels : sondes ponctuelles fixes en amont du navire.
La houle va vers +x ; le navire est en x = 0, les sondes en x < 0.
"""

import numpy as np

from waveforecast.physics import eta

# Zone des sondes en amont (m) : 300-600 m couvre un horizon de 30-60 s
# pour Tp ≈ 7-13 s (cf. physics.predictable_window).
X_FAR, X_NEAR = -600.0, -300.0


def probe_positions(n, x_far=X_FAR, x_near=X_NEAR):
    """n sondes réparties de x_far à x_near (une seule : au plus près)."""
    return np.linspace(x_far, x_near, n) if n > 1 else np.array([x_near])


def probes(sea, x_probes, t_start, t_end, fs=2.0, noise=0.03, seed=0):
    """
    Séries des sondes, échantillonnées à fs (Hz) sur [t_start, t_end[,
    avec un bruit gaussien d'écart-type noise·Hs.
    Retourne dict(t [Nt], x [Np], eta [Np, Nt], fs, noise).
    """
    t = np.arange(t_start, t_end, 1.0 / fs)
    x = np.asarray(x_probes, dtype=float)
    clean = eta(sea, x, t).T
    rng = np.random.default_rng(seed)
    data = clean + rng.normal(0.0, noise * sea['Hs'], clean.shape)
    return dict(t=t, x=x, eta=data.astype(np.float32), fs=float(fs), noise=float(noise))
