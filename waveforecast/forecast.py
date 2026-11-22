"""
Mode Prévision du jumeau : la mer devient une houle 1D connue (vers +x), des
sondes en amont du navire la mesurent, la LSQ la reconstruit, et deux navires
fantômes (modèle de Cummins) sont intégrés sur l'horizon :
  - « prévu »  : MODÈLE du navire (ce que croit le jumeau, cf. set_model)
                 et houle reconstruite par la LSQ à partir des sondes seules ;
  - « vérité » : navire RÉEL simulé et houle exacte (connue : mer synthétique).
L'écart cumule erreur de houle et erreur de modèle (corrigée par calibration).
Indépendant de Qt : appelé par main.py à chaque frame.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

import jonswap
from boat import HydroModel
from waveforecast import baseline_lsq, physics, sensors

PERIOD = 5.0        # s entre deux prévisions
HORIZON = 60.0      # s prévus
T_WIN = 180.0       # s de mesures des sondes par prévision
DT = 0.1            # pas des navires fantômes (RK4 ; périodes propres ≥ 2.5 s)
HISTORY = 30.0      # s de mouvement réel gardés pour l'affichage
NOISE = 0.03        # bruit des sondes / Hs
SEED = 42           # graine de la mer du mode Prévision

# Navire placé à 500 m dans la grille de 600 m, sondes 300-450 m en amont :
# tout reste dans la zone affichée. Horizon théorique ≈ 450 m / c_g,max
# (≈ 48 s à Tp = 10 s).
SHIP_X = 500.0
PROBE_OFFSETS = sensors.probe_positions(4, -450.0, -300.0)

# Période calme : mouvement prévu sous ces seuils sur les CALM_WINDOW
# prochaines secondes (limitées à la zone de prédictibilité)
CALM_WINDOW = 30.0
CALM_HEAVE = 0.5            # m
CALM_ROLL = np.radians(2.0)


@dataclass
class Forecast:
    t0: float
    times: np.ndarray       # [N] temps absolus
    predicted: np.ndarray   # [N, 12] états du fantôme « prévu »
    truth: np.ndarray       # [N, 12] états du fantôme « vérité »
    zone_end: float         # s après t0 : fin de la zone de prédictibilité
    calm: bool              # période calme prévue sur calm_window s
    calm_window: float

    def state_at(self, t):
        """État prévu interpolé au temps t (borné à l'horizon)."""
        return np.array([np.interp(t, self.times, c) for c in self.predicted.T])


def ghost_run(hydro, state, t0, zeta, dt=DT):
    """
    Intègre Cummins depuis `state` avec la houle zeta [N, 55] aux points de
    flottaison (un échantillon par pas). Retourne les états [N, 12].
    """
    waves = np.column_stack(hydro.wave_plane(zeta))
    out = np.empty((len(zeta), len(state)))
    out[0] = state
    for i in range(len(zeta) - 1):
        out[i + 1] = hydro.step_state(out[i], t0 + i * dt, dt, wave=tuple(waves[i]))
    return out


class ShipForecast:
    """Mode Prévision : fige la mer de `sim`, place le navire, prévoit toutes les PERIOD s."""

    def __init__(self, sim, seed=SEED):
        p = sim.sea.target          # état de mer visé (même en cours de transition)
        self.truth = physics.make_sea_1d(p['Hs'], p['Tp'], p['gamma'], seed)
        t = self.truth
        sim.sea.freeze(t['A'], t['omega'], t['k'], t['phi'], Hs=t['Hs'], Tp=t['Tp'],
                       gamma=t['gamma'])
        self.band = physics.energy_band(t['Tp'], t['gamma'], baseline_lsq.RECON_FRAC)
        self.x_probes = SHIP_X + PROBE_OFFSETS
        # Fin de zone vue du navire (bande 90 % d'énergie, comme l'évaluation)
        self.zone_end = physics.predictable_window(
            0.0, T_WIN, SHIP_X - self.x_probes, *physics.energy_band(t['Tp'], t['gamma']))[1]
        self.history = deque()
        self.last = None
        self.set_model(sim.params, sim)
        self.place(sim)

    def set_model(self, params, sim):
        """Caractéristiques que le jumeau attribue au navire (prévision « prévu »)."""
        self.model = HydroModel(params, sim.sea)

    def place(self, sim):
        """Navire en station en SHIP_X, au repos ; prévision relancée."""
        sim.state[:] = np.r_[SHIP_X, jonswap.Ly / 2, 0.0, 0.0, 0.0, sim.state[5], np.zeros(6)]
        sim.set_speed(0.0)
        self.history.clear()
        self.last = None
        self._next = sim.t

    def close(self, sim):
        sim.sea.unfreeze()

    def record(self, sim):
        """Mouvement réel (heave, roll, pitch) pour l'affichage de l'historique."""
        self.history.append((sim.t, *sim.state[2:5]))
        while self.history and self.history[0][0] < sim.t - HISTORY:
            self.history.popleft()

    def due(self, t):
        return t >= self._next

    def probe_elevations(self, t):
        """Élévation vraie aux sondes (affichage des bouées)."""
        return physics.eta(self.truth, self.x_probes, [t])[0]

    def start(self, sim):
        """Arguments figés de compute() pour l'état courant ; planifie la suivante.
        À appeler depuis la boucle principale (lit sim)."""
        self._next = sim.t + PERIOD
        self.model.U_target = sim.hydro.U_target
        return sim.t, sim.state.copy(), sim.hydro, self.model, sim.boat_xy()

    def compute(self, t0, state, real, model_hydro, xy):
        """
        Prévision depuis l'instant t0 (≈ 0.4 s de calcul). Ne lit ni ne modifie
        la simulation : peut tourner dans un thread pendant que le jumeau avance.
        """
        meas = sensors.probes(self.truth, self.x_probes, t0 - T_WIN, t0, noise=NOISE,
                              seed=int(t0 * 10))
        model = baseline_lsq.fit(meas['t'], meas['x'], meas['eta'], *self.band)
        times = t0 + np.arange(0.0, HORIZON + DT / 2, DT)
        # shortcut: points de flottaison figés à leur position en t0 — valable pour
        # un navire en station ou lent ; à suivre le long de la trajectoire sinon.
        xs, _ = real.waterplane_points(*xy, state[5])
        pred = ghost_run(model_hydro, state, t0, physics.eta(model, xs, times))
        true = ghost_run(real, state, t0, physics.eta(self.truth, xs, times))

        window = min(CALM_WINDOW, self.zone_end)
        k = times <= t0 + window
        calm = bool(np.abs(pred[k, 2]).max() < CALM_HEAVE
                    and np.abs(pred[k, 3]).max() < CALM_ROLL)
        return Forecast(t0, times, pred, true, self.zone_end, calm, window)

    def run(self, sim):
        """Prévision synchrone depuis l'état courant (tests, scripts)."""
        self.last = self.compute(*self.start(sim))
        return self.last
