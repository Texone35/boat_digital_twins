"""
boat.py — Modèle hydrodynamique 6 DDL basé sur les équations de Cummins (1962).

Équation de Cummins linéarisée (Ogilvie 1964) :
    [M + A(ω_e)] η̈ + B(ω_e) η̇ + C η = F_exc + F_prop

La fréquence de rencontre ω_e est calculée à chaque pas de temps
depuis la vitesse du navire et l'angle de rencontre avec la houle.
F_exc est calculée à partir de la mer simulée (jonswap.SeaState) sous la
flottaison : η (heave, roll, pitch) est donc une position ABSOLUE, par
rapport au niveau de mer calme.

Les caractéristiques du navire (BoatParams) et l'état de mer sont lus par
HydroModel : changer de bateau = construire un nouveau HydroModel ; les
paramètres de mer (Tp, direction) sont relus en direct à chaque pas.
"""

from dataclasses import dataclass, fields

import numpy as np
from matplotlib import colors as mcolors

import jonswap as _sea

# ────────────────────────────────────────────────
# CONSTANTES
# ────────────────────────────────────────────────
rho = 1025.0
g   = 9.81


# ────────────────────────────────────────────────
# CARACTÉRISTIQUES DU NAVIRE
# ────────────────────────────────────────────────
@dataclass
class BoatParams:
    name: str = 'Cargo'
    L: float = 100.0      # longueur à la flottaison (m)
    B: float = 20.0       # largeur (m)
    T: float = 8.0        # tirant d'eau (m)
    Cb: float = 0.61      # coefficient de bloc : déplacement ∇ = Cb·L·B·T
    Cwp: float = 0.85     # coefficient de flottaison : Awp = Cwp·L·B
    GM_T: float = 1.2     # hauteur métacentrique transversale (m)
    GM_L: float = 120.0   # hauteur métacentrique longitudinale (m)
    kxx: float = 0.35     # rayon de giration en roulis / B
    kyy: float = 0.25     # rayon de giration en tangage et lacet / L
    freeboard_ratio: float = 0.5   # franc-bord / tirant d'eau
    xf_ratio: float = 0.02         # centre de flottaison en avant du milieu / L
    speed: float = 2.5    # vitesse de consigne (m/s)
    color: str = '#FFD700'
    # Amortissement de roulis : linéaire (fraction ζ du critique) et
    # quadratique b_q (1/rad), moment −b_q·(Ixx + A44)·|p|·p (quilles de
    # roulis, viscosité). Paramètres que l'identification (shipid) retrouve.
    zeta_roll: float = 0.10
    roll_quad: float = 0.0

    # ── Grandeurs dérivées ──
    @property
    def nabla(self):      return self.Cb * self.L * self.B * self.T
    @property
    def mass(self):       return rho * self.nabla
    @property
    def Awp(self):        return self.Cwp * self.L * self.B
    @property
    def x_f(self):        return self.xf_ratio * self.L
    @property
    def freeboard(self):  return self.freeboard_ratio * self.T
    @property
    def Ixx(self):        return self.mass * (self.kxx * self.B)**2
    @property
    def Iyy(self):        return self.mass * (self.kyy * self.L)**2
    @property
    def Izz(self):        return self.Iyy

    def validate(self):
        """Liste des incohérences (vide si les caractéristiques sont valides)."""
        err = []
        if not (5.0 <= self.L <= 400.0):
            err.append('L doit être entre 5 et 400 m (domaine de mer de 600 m).')
        if not (0 < self.B < self.L):
            err.append('Il faut 0 < B < L.')
        if not (0 < self.T < self.B):
            err.append('Il faut 0 < T < B.')
        if not (0.35 <= self.Cb <= 0.9):
            err.append('Cb doit être entre 0.35 et 0.90.')
        if not (self.Cb <= self.Cwp <= 1.0):
            err.append('Il faut Cb ≤ Cwp ≤ 1.')
        if self.GM_T <= 0 or self.GM_L <= 0:
            err.append('GM_T et GM_L doivent être > 0 (navire instable sinon).')
        if not (0.2 <= self.kxx <= 0.5) or not (0.15 <= self.kyy <= 0.35):
            err.append('Rayons de giration hors plage (kxx 0.2-0.5, kyy 0.15-0.35).')
        if not (0 < self.zeta_roll <= 0.5) or self.roll_quad < 0:
            err.append('Amortissement de roulis : 0 < ζ ≤ 0.5 et b_q ≥ 0.')
        if self.freeboard_ratio <= 0:
            err.append('Le franc-bord doit être > 0.')
        try:
            mcolors.to_rgb(self.color)
        except ValueError:
            err.append(f'Couleur invalide : {self.color}')
        return err

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ────────────────────────────────────────────────
# FRÉQUENCE DE RENCONTRE
# ────────────────────────────────────────────────
def encounter_frequency(omega, V, mu):
    """
    ω_e = ω − (ω²/g)·V·cos(μ)
    μ = 0   : houle de l'arrière (même sens que le navire)
    μ = π/2 : houle de travers
    μ = π   : houle de face
    Clampé à [ω/5, 2ω] pour éviter singularités.
    """
    we = omega - (omega**2 / g) * V * np.cos(mu)
    return float(np.clip(we, omega / 5.0, 2.0 * omega))

def heading_angle(psi, wave_hdg):
    """μ ∈ [0, π] — angle entre cap navire et direction de propagation de la houle."""
    mu = (wave_hdg - psi) % (2 * np.pi)
    return mu if mu <= np.pi else 2*np.pi - mu

def heading_label(mu):
    """Nom usuel de l'incidence de houle μ (rad)."""
    deg = np.degrees(mu)
    if deg < 30:   return "houle de l'arrière"
    if deg < 60:   return 'houle de hanche'
    if deg < 120:  return 'houle de travers'
    if deg < 150:  return 'houle de joue'
    return 'houle de face'


# ────────────────────────────────────────────────
# MODÈLE HYDRODYNAMIQUE
# ────────────────────────────────────────────────
# Fraction d'amortissement critique ζ cible par DDL (valeurs navire typiques)
_ZETA_TARGET = np.array([0.05, 0.08, 0.10, 0.10, 0.08, 0.05])
# Plage de validité des coefficients hydrodynamiques (rad/s)
_OMEGA_MIN, _OMEGA_MAX = 0.20, 2.00


class HydroModel:
    """
    Coefficients et intégration de Cummins pour un navire donné dans une
    mer donnée. Les paramètres de mer (ω_p, direction) sont relus sur
    `sea` à chaque appel : ils peuvent changer en cours de simulation.

    Coefficients hydrodynamiques (Salvesen, Tuck & Faltinsen 1970, simplifiés) :
      A_ii(ω) = A0_ii · clip(1 + 0.15 (ω/ω_p − 1), 0.8, 1.2)   (±15 % autour de ω_p)
      B_ii(ω) = B0_ii · √(ω/ω_n)                                 (radiatif, eaux profondes)
    avec B0_ii = 2 ζ_i √(C_ii (M_ii + A_ii(ω_n))).
    """

    def __init__(self, params, sea):
        self.params = params
        self.sea = sea
        self.U_target = params.speed
        p = params
        L, B, T = p.L, p.B, p.T
        self.m = m = p.mass

        # ── Raideur hydrostatique (indépendante de ω) ──
        C = np.zeros((6, 6))
        C[2, 2] =  rho * g * p.Awp
        C[3, 3] =  rho * g * p.nabla * p.GM_T
        C[4, 4] =  rho * g * p.nabla * p.GM_L
        C[2, 4] = -rho * g * p.Awp * p.x_f
        C[4, 2] =  C[2, 4]
        self.C_mat = C

        # ── Masses ajoutées à ω_p ──
        self.A0 = np.array([0.10 * rho * L * B * T,
                            0.80 * rho * L * B * T,
                            0.20 * rho * L * B * T,
                            0.40 * rho * L * B**3 / 12,
                            0.80 * rho * L**3 * B / 12,
                            0.20 * rho * L * B**3 / 12])
        self.M_diag = np.array([m, m, m, p.Ixx, p.Iyy, p.Izz])
        # Couplages sway-roll et heave-pitch de la matrice de masse ajoutée
        self._A24 = 0.05 * m
        self._A35 = 0.10 * m

        # Raideur effective pour l'amortissement : pseudo-raideur pour les
        # DDL sans rappel hydrostatique (surge, sway, yaw)
        c_pseudo = rho * g * p.Awp * 0.01
        self._C_eff = np.array([c_pseudo, c_pseudo, C[2, 2], C[3, 3], C[4, 4], c_pseudo])

        # ── Amortissement radiatif ──
        self.omega_n = self._omega_n()
        MAn = self.M_diag + self.A_diag(self.omega_n)
        zeta = _ZETA_TARGET.copy()
        zeta[3] = p.zeta_roll
        self.B0 = zeta * 2.0 * np.sqrt(self._C_eff * MAn)

        # Points d'échantillonnage de la flottaison L×B (repère navire). Grille
        # régulière centrée : les colonnes x, y sont orthogonales, donc le plan
        # des moindres carrés se calcule directement (moyenne + deux projections).
        self._wp_x, self._wp_y = [a.ravel() for a in np.meshgrid(
            np.linspace(-L/2, L/2, 11), np.linspace(-B/2, B/2, 5), indexing='ij')]

    # ── Coefficients dépendant de ω ──
    @property
    def wave_heading(self):
        return self.sea.theta_mean

    def A_diag(self, we):
        """Masses ajoutées diagonales à ω_e (scalaire, ou une valeur par DDL)."""
        we = np.clip(we, _OMEGA_MIN, _OMEGA_MAX)
        return self.A0 * np.clip(1.0 + 0.15 * (we / self.sea.omega_p - 1.0), 0.80, 1.20)

    def B_diag(self, we):
        we = np.clip(we, _OMEGA_MIN, _OMEGA_MAX)
        return self.B0 * np.sqrt(we / self.omega_n)

    def _omega_n(self):
        """Fréquences propres des 6 DDL (masses ajoutées prises à ω_p)."""
        return np.sqrt(self._C_eff / (self.M_diag + self.A_diag(self.sea.omega_p)))

    def natural_periods(self):
        """Périodes propres (s) : {'heave', 'roll', 'pitch'} à l'état de mer courant."""
        Tn = 2 * np.pi / self._omega_n()
        return dict(heave=Tn[2], roll=Tn[3], pitch=Tn[4])

    def get_MA_matrix(self, omega_e):
        MA = np.diag(self.M_diag + self.A_diag(omega_e))
        MA[1, 3] = MA[3, 1] = self._A24
        MA[2, 4] = MA[4, 2] = self._A35
        return MA

    def get_B_matrix(self, omega_e):
        return np.diag(self.B_diag(omega_e))

    def encounter(self, state):
        """(μ, V, ω_e) pour l'état courant."""
        mu = heading_angle(state[5], self.wave_heading)
        V  = abs(state[6])
        return mu, V, encounter_frequency(self.sea.omega_p, V, mu)

    # ── Houle sous la flottaison — lien avec la mer simulée ──
    def waterplane_wave(self, Z, x, y, psi):
        """
        Houle vue par la flottaison du navire en (x, y), cap psi : plan des
        moindres carrés  ζ ≈ ζ̄ + s_x·x_b + s_y·y_b  ajusté sur la surface Z.

        Retourne (ζ̄, s_x, s_y) : élévation moyenne sous la flottaison et pentes
        longitudinale / transversale dans le repère navire. Moyenner sur toute
        la flottaison filtre naturellement les vagues courtes devant L.
        """
        xs, ys = self.waterplane_points(x, y, psi)
        return self.wave_plane(_sea.interpolate_sea_at_batch(Z, xs % _sea.Lx, ys % _sea.Ly))

    def waterplane_points(self, x, y, psi):
        """Points d'échantillonnage de la flottaison (11 × 5), repère terrestre."""
        c, s = np.cos(psi), np.sin(psi)
        return (x + c*self._wp_x - s*self._wp_y, y + s*self._wp_x + c*self._wp_y)

    def wave_plane(self, zeta):
        """
        Plan des moindres carrés (ζ̄, s_x, s_y) sur les élévations zeta aux
        points de waterplane_points ; zeta [..., 55] → trois tableaux [...].
        """
        wx, wy = self._wp_x, self._wp_y
        return zeta.mean(axis=-1), zeta @ wx / (wx @ wx), zeta @ wy / (wy @ wy)

    # ── Forces d'excitation de houle — Froude-Krylov hydrostatique ──
    def excitation_forces(self, wave):
        """
        F_exc = C · η_houle

        η_houle = position (heave, roll, pitch) d'un navire qui épouserait
        exactement le plan de houle sous sa flottaison. La raideur C tire le
        navire vers cette position ; masse ajoutée et amortissement B(ω_e)
        introduisent le retard de phase et atténuent la réponse aux vagues
        rapides. Surge, sway et yaw ne sont pas excités.

        Conventions de signe identiques au rendu (Boat.compute_mesh) :
          roll  > 0 : bord y > 0 relevé  → roll  = +atan(s_y)
          pitch > 0 : proue x > 0 baissée → pitch = −atan(s_x)
        """
        zeta, sx, sy = wave
        eta_wave = np.zeros(6)
        eta_wave[2] = zeta
        eta_wave[3] = np.arctan(sy)
        eta_wave[4] = -np.arctan(sx)
        return self.C_mat @ eta_wave

    # ── Matrice de Coriolis linéarisée ──
    def coriolis_matrix(self, nu, omega_e):
        mA22 = self.m + self.A_diag(omega_e)[1]
        C_cor = np.zeros((6, 6))
        u = nu[0]
        C_cor[1, 5] = -mA22 * u
        C_cor[5, 1] =  mA22 * u
        return C_cor

    # ── Dérivée d'état — Cummins linéarisé ──
    def state_derivative(self, t, state, wave=(0.0, 0.0, 0.0)):
        """
        [M + A(ω_e)] η̈ + B(ω_e) η̇ + C η = F_exc + F_prop

        state = [η(6), η̇(6)]
        wave  = (ζ̄, s_x, s_y) sous la flottaison, cf. waterplane_wave()
        """
        eta = state[:6]
        nu  = state[6:]

        psi = eta[5]
        _, _, omega_e = self.encounter(state)

        MA = self.get_MA_matrix(omega_e)
        Bd = self.B_diag(omega_e)

        rhs = (- self.C_mat @ eta
               - Bd * nu
               - self.coriolis_matrix(nu, omega_e) @ nu
               + self.excitation_forces(wave))
        # Propulsion : poussée qui équilibre la résistance B11·u à la vitesse
        # de consigne → u converge vers U_target en régime établi
        rhs[0] += Bd[0] * self.U_target
        # Amortissement quadratique de roulis (nul par défaut)
        rhs[3] -= self.params.roll_quad * MA[3, 3] * abs(nu[3]) * nu[3]

        nu_dot = np.linalg.solve(MA, rhs)

        # Rotation repère navire → repère terrestre pour la position (x,y) uniquement :
        # sans cela, surge/sway s'intègrent comme des vitesses globales et la trajectoire
        # ne suit plus le cap dès que le yaw dérive. Les autres DDL (heave/roll/pitch/yaw)
        # gardent l'approximation petits angles usuelle (eta_dot = nu).
        cpsi, spsi = np.cos(psi), np.sin(psi)
        eta_dot = nu.copy()
        eta_dot[0] = nu[0] * cpsi - nu[1] * spsi
        eta_dot[1] = nu[0] * spsi + nu[1] * cpsi

        return np.concatenate([eta_dot, nu_dot])

    # ── Intégration RK4 ──
    def step_state(self, state, t, dt, wave=(0.0, 0.0, 0.0)):
        # Houle figée sur le pas dt (échantillonnée une fois par frame)
        kw = dict(wave=wave)
        k1 = self.state_derivative(t,          state,              **kw)
        k2 = self.state_derivative(t+0.5*dt,   state+0.5*dt*k1,   **kw)
        k3 = self.state_derivative(t+0.5*dt,   state+0.5*dt*k2,   **kw)
        k4 = self.state_derivative(t+dt,       state+dt*k3,        **kw)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ────────────────────────────────────────────────
# GÉOMÉTRIE & RENDU DU BATEAU
# ────────────────────────────────────────────────
def hull_exponent(Cb):
    """
    Exposant des profils de carène en fonction du coefficient de bloc :
    ≈ 4 pour un cargo (Cb ≈ 0.6, murailles droites, fond plat), plus faible
    pour une coque fine (sections en V), plus élevé pour une coque pleine.
    """
    return float(np.clip(4.0 + 12.0 * (Cb - 0.6), 2.0, 8.0))


def create_boat_mesh(length, width, draft, freeboard, Cb=0.6, Nu=25, Nv=9):
    """
    Coque FERMÉE : carène + pont plat. Le plan Z=0 (local) est la ligne de
    flottaison de conception : la quille (milieu) est à −draft, le livet
    (bord du pont) à +freeboard.

    - Vue en plan : étrave en pointe (x > 0), arrière légèrement affiné.
    - Sections pleines (profil en |v|^p) : murailles quasi verticales, fond
      plat au milieu ; profil longitudinal en |u|^p : corps cylindrique au
      centre, fonds qui remontent vers l'étrave et l'arrière. p dépend de Cb
      (hull_exponent). Nu et Nv fixent la topologie, identique pour tous
      les navires.

    Retourne (verts, tris, is_deck) — is_deck[i] vrai pour les triangles du
    pont (couleur propre dans Boat).
    """
    p = hull_exponent(Cb)
    xi = np.linspace(-1.0, 1.0, Nu)          # abscisse normalisée (poupe → étrave)
    # Position transversale normalisée, resserrée vers les murailles (où la
    # profondeur varie vite) : sinon un seul triangle couvrirait le flanc du
    # livet jusqu'à mi-profondeur, à cheval sur la flottaison.
    eta = np.sin(0.5 * np.pi * np.linspace(-1.0, 1.0, Nv))
    XI, ETA = np.meshgrid(xi, eta, indexing='ij')

    bow   = np.clip((XI - 0.35) / 0.65, 0, 1)
    stern = np.clip((-XI - 0.70) / 0.30, 0, 1)
    half_w = 0.5 * width * np.sqrt(1 - bow**2) * (1 - 0.3 * stern**2)

    U = 0.5 * length * XI
    V = ETA * half_w
    depth = (1 - np.abs(XI)**p) * (1 - np.abs(ETA)**p)
    Z_hull = freeboard - (draft + freeboard) * depth

    n = Nu * Nv
    verts = np.vstack([
        np.column_stack([U.ravel(), V.ravel(), Z_hull.ravel()]),
        np.column_stack([U.ravel(), V.ravel(), np.full(n, freeboard)]),
    ])
    idx = (np.arange(Nu - 1)[:, None] * Nv + np.arange(Nv - 1)[None, :]).ravel()
    quads_a = np.column_stack([idx, idx + 1, idx + Nv])
    quads_b = np.column_stack([idx + 1, idx + Nv + 1, idx + Nv])
    hull = np.vstack([quads_a, quads_b])
    deck = hull[:, ::-1] + n                 # même grille, orientation inversée
    tris = np.vstack([hull, deck])
    is_deck = np.r_[np.zeros(len(hull), bool), np.ones(len(deck), bool)]
    return verts, tris, is_deck


class Boat:
    """Géométrie et couleurs de la coque d'un navire décrit par BoatParams."""

    def __init__(self, params, Nu=41, Nv=15,
                 underwater_color='#2E8B57', deck_color='#9C7A54'):
        self.params = params
        self.length, self.width = params.L, params.B
        self.draft, self.freeboard = params.T, params.freeboard
        self.local_vertices, self.triangles, self._is_deck = create_boat_mesh(
            params.L, params.B, params.T, params.freeboard, Cb=params.Cb,
            Nu=Nu, Nv=Nv)

        def _valid(c, fallback):
            try:
                mcolors.to_rgb(c); return c
            except ValueError:
                return fallback
        self.color = _valid(params.color, '#FFD700')
        self.underwater_color = _valid(underwater_color, '#2E8B57')
        self.deck_color = _valid(deck_color, '#9C7A54')
        # Couleur hors de l'eau : pont ou coque
        self._dry_colors = np.where(self._is_deck, self.deck_color, self.color)

        # Coloration de repli (ligne de flottaison de conception, Z local=0)
        # utilisée quand compute_mesh() ne reçoit pas de fonction d'élévation
        # de mer (sea_elev_fn). Un triangle n'est classé "immergé" que si ses
        # 3 sommets sont sous le seuil : un critère sur la moyenne des sommets
        # laisse une frontière ambiguë (mouchetée), où deux triangles voisins
        # d'un même carreau peuvent basculer chacun d'un côté différent.
        tri_local_z = self.local_vertices[self.triangles][:, :, 2]
        fully_submerged = np.all(tri_local_z < 0, axis=1)
        self._tri_colors = np.where(fully_submerged,
                                    self.underwater_color, self._dry_colors)
        self.world_vertices = self.local_vertices

    def compute_mesh(self, pitch, roll, x_pos, y_pos, z_center, yaw=0.0,
                     sea_elev_fn=None):
        """
        Triangles de la coque en position monde et leur couleur.

        sea_elev_fn(x_arr, y_arr) -> élévation locale de la mer (tableau),
        si fournie : la coloration immergé/émergé est recalculée à cette
        frame à partir de la vraie surface de mer sous chaque sommet de la
        coque (immersion dynamique), plutôt que d'utiliser le seuil de
        conception statique (self._tri_colors).
        """
        Rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        Ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        verts=(Rz@Ry@Rx@self.local_vertices.T).T+np.array([x_pos,y_pos,z_center])
        self.world_vertices = verts   # sommets partagés (indexés par self.triangles)
        tri_verts = verts[self.triangles]

        if sea_elev_fn is not None:
            local_elev = sea_elev_fn(verts[:,0], verts[:,1])
            vertex_submerged = verts[:,2] < local_elev
            tri_submerged = np.all(vertex_submerged[self.triangles], axis=1)
            tri_colors = np.where(tri_submerged, self.underwater_color,
                                  self._dry_colors)
        else:
            tri_colors = self._tri_colors

        return tri_verts, tri_colors
