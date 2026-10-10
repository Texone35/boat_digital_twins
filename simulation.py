"""
Simulation couplée mer → navire, indépendante de l'affichage.

Couplage mer → bateau à chaque frame :
  - élévation η(x_b,y_b,t)     → élévation moyenne sous la flottaison
  - pentes ∂η/∂x, ∂η/∂y       → moments Froude-Krylov roll/pitch

Position du bateau : heave, roll, pitch (eta[2:5]) sont ABSOLUS, par
rapport au niveau de mer calme.
"""

import numpy as np

import jonswap as sea_mod
from boat import Boat, HydroModel


class Simulation:
    """
    État du navire et pas de temps.
    state = [η(6), ν(6)] avec
      η = [surge(x), sway(y), heave, roll, pitch, yaw]
      ν = [u,        v,       w,     p,    q,     r  ]
    """

    def __init__(self, sea, params, dt=0.05):
        self.sea = sea
        self.dt = dt
        self.t = 0.0
        self.state = np.zeros(12, dtype=float)
        self.state[0] = sea_mod.Lx / 2
        self.state[1] = sea_mod.Ly / 2
        self.state[6] = params.speed
        self.set_boat(params)

    def boat_xy(self):
        """Position du bateau repliée dans le domaine périodique de la mer."""
        return self.state[0] % sea_mod.Lx, self.state[1] % sea_mod.Ly

    # ── Pilotage en direct ──
    def set_boat(self, params):
        """
        Remplace le navire à chaud : position, cap et vitesse d'avance
        conservés ; heave/roll/pitch et autres vitesses remis à zéro pour
        éviter un saut violent (le nouveau navire part de son équilibre).
        """
        self.params = params
        self.hydro = HydroModel(params, self.sea)
        self.boat = Boat(params)
        self.state[2:5] = 0.0
        self.state[7:12] = 0.0

    def set_speed(self, U):
        self.hydro.U_target = float(U)

    def set_heading(self, psi):
        """Cap imposé (rad). Stable : pas de raideur en lacet (C66 = 0)."""
        self.state[5] = float(psi)
        self.state[11] = 0.0

    def reset(self):
        """Bateau recentré, au repos, à la vitesse de consigne."""
        U, psi = self.hydro.U_target, self.state[5]
        self.state[:] = 0.0
        self.state[0], self.state[1] = sea_mod.Lx / 2, sea_mod.Ly / 2
        self.state[5], self.state[6] = psi, U

    # ── Pas de temps ──
    def step(self, n_sub=1):
        """
        Avance de n_sub pas de dt (rattrapage du temps réel). La houle sous
        la flottaison est échantillonnée une fois, au temps final, et
        réutilisée pour les sous-pas. Retourne (Z, Dx, Dy, wave).
        """
        dt = self.dt
        t_new = self.t + n_sub * dt
        xb, yb = self.boat_xy()

        # 1. Transition éventuelle de l'état de mer, continue sous le bateau
        self.sea.advance(n_sub * dt, ref=(xb, yb))
        # 2. Surface de mer (Z : élévation pour la physique ; Dx, Dy :
        # déplacements horizontaux pour l'affichage)
        Z, Dx, Dy = self.sea.generate_fields(t_new)
        # 3. Houle sous la flottaison : élévation moyenne + pentes
        wave = self.hydro.waterplane_wave(Z, xb, yb, self.state[5])
        # 4. Intégration Cummins RK4
        for k in range(n_sub):
            self.state[:] = self.hydro.step_state(self.state, self.t + k * dt, dt, wave=wave)
        self.t = t_new
        return Z, Dx, Dy, wave

    def fields(self):
        """Surface au temps courant sans avancer (affichage en pause)."""
        Z, Dx, Dy = self.sea.generate_fields(self.t)
        xb, yb = self.boat_xy()
        return Z, Dx, Dy, self.hydro.waterplane_wave(Z, xb, yb, self.state[5])
