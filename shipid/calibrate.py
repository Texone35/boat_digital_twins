"""
Calibration du modèle du jumeau sur le roulis du navire réel, et
surveillance glissante (une identification toutes les PERIOD s sur les
WINDOW dernières secondes). Indépendant de Qt.
"""

from collections import deque
from dataclasses import replace

import numpy as np

from shipid import identify, records

WINDOW = 300.0       # s de roulis utilisés par identification
MIN_WINDOW = 120.0   # s minimum avant la première identification
PERIOD = 20.0        # s entre deux identifications
FS = 10.0            # Hz : rééchantillonnage uniforme des mesures


def calibrated_params(model, sea, ident):
    """
    BoatParams du modèle recalé : GM_T déduit de ω_n (inertie du modèle
    supposée connue), zeta_roll choisi pour reproduire le ζ identifié
    (ζ effectif est proportionnel à zeta_roll), roll_quad = b_q.
    """
    gm = records.gm_from_omega(ident['omega_n'], model, sea)
    zeta_per_unit = records.roll_truth(replace(model, GM_T=gm, zeta_roll=1.0), sea)['zeta']
    return replace(model, GM_T=round(gm, 3),
                   zeta_roll=round(float(np.clip(ident['zeta'] / zeta_per_unit, 0.005, 0.5)), 4),
                   roll_quad=round(max(ident['b_q'], 0.0), 3))


def hidden_ship(model, seed=None):
    """Navire « réel » tiré au hasard autour du modèle : GM_T plus faible
    (chargement), amortissements de roulis différents."""
    rng = np.random.default_rng(seed)
    return replace(model, name=f'{model.name} (réel caché)',
                   GM_T=round(model.GM_T * rng.uniform(0.6, 0.95), 3),
                   zeta_roll=round(rng.uniform(0.03, 0.08), 3),
                   roll_quad=round(rng.uniform(0.5, 2.0), 2))


def identify_roll(t, roll, phi_w):
    """Erreur d'équation (initialisation) puis erreur de sortie."""
    return identify.output_error(t, roll, phi_w, identify.equation_error(t, roll, phi_w))


class RollMonitor:
    """Tampon du roulis mesuré et de la pente de houle, pour l'identification glissante."""

    def __init__(self):
        self.buf = deque()
        self._next = 0.0

    def clear(self):
        self.buf.clear()

    def record(self, t, roll, phi_w):
        self.buf.append((t, roll, phi_w))
        while self.buf and self.buf[0][0] < t - WINDOW:
            self.buf.popleft()

    def duration(self):
        return self.buf[-1][0] - self.buf[0][0] if len(self.buf) > 1 else 0.0

    def ready(self, t):
        return self.duration() >= MIN_WINDOW and t >= self._next

    def start(self, t):
        """Fenêtre rééchantillonnée à FS (t, roll, phi_w) ; planifie la suivante."""
        self._next = t + PERIOD
        a = np.array(self.buf)
        tu = np.arange(a[0, 0], a[-1, 0], 1.0 / FS)
        return tu, np.interp(tu, a[:, 0], a[:, 1]), np.interp(tu, a[:, 0], a[:, 2])
