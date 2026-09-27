"""
boat.py — Modèle hydrodynamique 6 DDL basé sur les équations de Cummins (1962).

Équation de Cummins linéarisée (Ogilvie 1964) :
    [M + A(ω_e)] η̈ + B(ω_e) η̇ + C η = F_exc

La fréquence de rencontre ω_e est calculée à chaque pas de temps
depuis la vitesse du navire et l'angle de rencontre avec la houle.
F_exc est calculée à partir de la mer simulée (jonswap.py) sous la
flottaison : η (heave, roll, pitch) est donc une position ABSOLUE, par
rapport au niveau de mer calme.
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import jonswap as _sea

# ────────────────────────────────────────────────
# CONSTANTES & GÉOMÉTRIE
# ────────────────────────────────────────────────
rho = 1025.0
g   = 9.81

L   = 100.0
B   =  20.0
T   =   8.0
m   = 10_000_000.0
nabla = m / rho
Awp   = L * B
x_f   = 2.0

GM_T = 1.2
GM_L = 8.0

Ixx = 0.40 * m * (B/2)**2
Iyy = m * (L**2 + B**2) / 12
Izz = m * (L**2 + B**2) / 12

# ────────────────────────────────────────────────
# RAIDEUR HYDROSTATIQUE (indépendante de ω)
# ────────────────────────────────────────────────
C_mat = np.zeros((6, 6))
C_mat[2, 2] =  rho * g * Awp
C_mat[3, 3] =  rho * g * nabla * GM_T
C_mat[4, 4] =  rho * g * nabla * GM_L
C_mat[2, 4] = -rho * g * Awp * x_f
C_mat[4, 2] =  C_mat[2, 4]

# ────────────────────────────────────────────────
# PARAMÈTRES DE HOULE — synchronisés avec jonswap.py
# (même Hs/Tp/direction que la mer réellement générée et affichée)
# ────────────────────────────────────────────────
Hs           = _sea.Hs
Tp           = _sea.Tp
omega_p      = 2 * np.pi / Tp
wave_heading = _sea.theta_mean

# ────────────────────────────────────────────────
# TABLES DES COEFFICIENTS HYDRODYNAMIQUES A(ω), B(ω)
# Référence : Salvesen, Tuck & Faltinsen (1970)
# ────────────────────────────────────────────────
_omega_tab = np.array([0.20, 0.40, 0.60, omega_p, 1.00, 1.20, 1.50, 2.00])

# ── Masses ajoutées A_ii(ω) ─────────────────────
def _build_added_mass():
    A0 = {
        'A11': 0.10 * rho * L * B * T,
        'A22': 0.80 * rho * L * B * T,
        'A33': 0.20 * rho * L * B * T,
        'A44': 0.40 * rho * L * B**3 / 12,
        'A55': 0.80 * rho * L**3 * B / 12,
        'A66': 0.20 * rho * L * B**3 / 12,
    }
    tables = {}
    for k, a in A0.items():
        # Variation ±15 % autour de ω_p (eaux profondes, navire élancé)
        fac = np.clip(1.0 + 0.15 * (_omega_tab / omega_p - 1.0), 0.80, 1.20)
        tables[k] = a * fac
    return tables

_A_tab  = _build_added_mass()
_A_itp  = {k: interp1d(_omega_tab, v, kind='linear',
                        bounds_error=False, fill_value=(v[0], v[-1]))
           for k, v in _A_tab.items()}

def _A(key, we):
    return float(_A_itp[key](np.clip(we, _omega_tab[0], _omega_tab[-1])))

# ── Périodes propres (pour le calcul de l'amortissement cible) ───
def _omega_n(dof):
    """Fréquence propre du DDL dof (0-5) à ω_p."""
    C_diag = [0, 0, C_mat[2,2], C_mat[3,3], C_mat[4,4], 0]
    M_diag = [m, m, m, Ixx, Iyy, Izz]
    A_keys = ['A11','A22','A33','A44','A55','A66']
    C_i = C_diag[dof] if C_diag[dof] > 0 else rho*g*Awp*0.01
    MA_i = M_diag[dof] + _A(A_keys[dof], omega_p)
    return np.sqrt(C_i / MA_i)

_omega_n_all = np.array([_omega_n(i) for i in range(6)])

# ── Amortissement radiatif B_ii(ω) ──────────────
# Fraction d'amortissement critique ζ cible par DDL (valeurs navire typiques)
_ZETA_TARGET = np.array([0.05, 0.08, 0.10, 0.10, 0.08, 0.05])

def _build_damping():
    """
    B_ii(ω_n) = 2 ζ √(C_ii · (M_ii + A_ii(ω_n)))
    Variation fréquentielle : B ∝ √ω  (comportement radiatif, eaux profondes)
    """
    A_keys = ['A11','A22','A33','A44','A55','A66']
    C_diag = [rho*g*Awp*0.01, rho*g*Awp*0.01,
              C_mat[2,2], C_mat[3,3], C_mat[4,4], rho*g*Awp*0.01]
    M_diag = [m, m, m, Ixx, Iyy, Izz]

    tables = {}
    keys   = ['B11','B22','B33','B44','B55','B66']
    for i, key in enumerate(keys):
        wn  = _omega_n_all[i]
        MAn = M_diag[i] + _A(A_keys[i], wn)
        Bc  = 2.0 * np.sqrt(C_diag[i] * MAn)
        B0  = _ZETA_TARGET[i] * Bc         # B à ω_n
        # Variation √ω normalisée à ω_n
        fac = np.sqrt(np.maximum(_omega_tab / wn, 0.0))
        tables[key] = B0 * fac
    return tables

_B_tab = _build_damping()
_B_itp = {k: interp1d(_omega_tab, v, kind='linear',
                       bounds_error=False, fill_value=(v[0], v[-1]))
          for k, v in _B_tab.items()}

def _B(key, we):
    return float(_B_itp[key](np.clip(we, _omega_tab[0], _omega_tab[-1])))

# ── Matrices M+A(ω_e) et B(ω_e) ─────────────────
def get_MA_matrix(omega_e):
    A24 = 0.05 * m;  A42 = A24
    A35 = 0.10 * m;  A53 = A35
    return np.array([
        [m+_A('A11',omega_e), 0,                   0,                  0,                   0,                   0                  ],
        [0,                   m+_A('A22',omega_e),  0,                  A24,                 0,                   0                  ],
        [0,                   0,                   m+_A('A33',omega_e), 0,                   A35,                 0                  ],
        [0,                   A42,                 0,                  Ixx+_A('A44',omega_e),0,                   0                  ],
        [0,                   0,                   A53,                0,                   Iyy+_A('A55',omega_e),0                  ],
        [0,                   0,                   0,                  0,                   0,                   Izz+_A('A66',omega_e)],
    ])

def get_B_matrix(omega_e):
    return np.diag([_B(k, omega_e) for k in
                    ['B11','B22','B33','B44','B55','B66']])

# ────────────────────────────────────────────────
# FRÉQUENCE DE RENCONTRE
# ────────────────────────────────────────────────
def encounter_frequency(omega, V, mu):
    """
    ω_e = ω − (ω²/g)·V·cos(μ)
    μ = 0   : houle de face (head seas)
    μ = π/2 : houle de travers (beam seas)
    μ = π   : houle de dos (following seas)
    Clampé à [ω_p/5, 2ω] pour éviter singularités.
    """
    we = omega - (omega**2 / g) * V * np.cos(mu)
    return float(np.clip(we, omega_p / 5.0, 2.0 * omega))

def heading_angle(psi, wave_hdg):
    """μ ∈ [0, π] — angle entre cap navire et direction de houle."""
    mu = (wave_hdg - psi) % (2 * np.pi)
    return mu if mu <= np.pi else 2*np.pi - mu

# ────────────────────────────────────────────────
# HOULE SOUS LA FLOTTAISON — lien avec la mer simulée (jonswap.py)
# ────────────────────────────────────────────────
# Points d'échantillonnage de la flottaison L×B (repère navire). Grille
# régulière centrée : les colonnes x, y sont orthogonales, donc le plan des
# moindres carrés se calcule directement (moyenne + deux projections).
_wp_x, _wp_y = [a.ravel() for a in np.meshgrid(
    np.linspace(-L/2, L/2, 11), np.linspace(-B/2, B/2, 5), indexing='ij')]

def waterplane_wave(Z, x, y, psi):
    """
    Houle vue par la flottaison du navire en (x, y), cap psi : plan des
    moindres carrés  ζ ≈ ζ̄ + s_x·x_b + s_y·y_b  ajusté sur la surface Z.

    Retourne (ζ̄, s_x, s_y) : élévation moyenne sous la flottaison et pentes
    longitudinale / transversale dans le repère navire. Moyenner sur toute
    la flottaison filtre naturellement les vagues courtes devant L.
    """
    c, s = np.cos(psi), np.sin(psi)
    xs = x + c*_wp_x - s*_wp_y
    ys = y + s*_wp_x + c*_wp_y
    zeta = _sea.interpolate_sea_at_batch(Z, xs % _sea.Lx, ys % _sea.Ly)
    return (zeta.mean(),
            np.dot(_wp_x, zeta) / np.dot(_wp_x, _wp_x),
            np.dot(_wp_y, zeta) / np.dot(_wp_y, _wp_y))

# ────────────────────────────────────────────────
# FORCES D'EXCITATION DE HOULE — Froude-Krylov hydrostatique
# ────────────────────────────────────────────────
def excitation_forces(wave):
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
    return C_mat @ eta_wave

# ────────────────────────────────────────────────
# MATRICE DE CORIOLIS LINÉARISÉE
# ────────────────────────────────────────────────
def coriolis_matrix(nu, omega_e):
    A22 = _A('A22', omega_e)
    C_cor = np.zeros((6, 6))
    u = nu[0]
    C_cor[1, 5] = -(m + A22) * u
    C_cor[5, 1] =  (m + A22) * u
    return C_cor

# ────────────────────────────────────────────────
# DÉRIVÉE D'ÉTAT — CUMMINS LINÉARISÉ
# ────────────────────────────────────────────────
def state_derivative(t, state, wave=(0.0, 0.0, 0.0)):
    """
    [M + A(ω_e)] η̈ + B(ω_e) η̇ + C η = F_exc

    state = [η(6), η̇(6)]
    wave  = (ζ̄, s_x, s_y) sous la flottaison, cf. waterplane_wave()
    """
    eta = state[:6]
    nu  = state[6:]

    psi     = eta[5]
    mu      = heading_angle(psi, wave_heading)
    V       = abs(nu[0])
    omega_e = encounter_frequency(omega_p, V, mu)

    MA  = get_MA_matrix(omega_e)
    Bd  = get_B_matrix(omega_e)

    rhs = (- C_mat @ eta
           - Bd @ nu
           - coriolis_matrix(nu, omega_e) @ nu
           + excitation_forces(wave))

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

# ────────────────────────────────────────────────
# INTÉGRATION RK4
# ────────────────────────────────────────────────
def step_state(state, t, dt, wave=(0.0, 0.0, 0.0)):
    # Houle figée sur le pas dt (échantillonnée une fois par frame)
    kw = dict(wave=wave)
    k1 = state_derivative(t,          state,              **kw)
    k2 = state_derivative(t+0.5*dt,   state+0.5*dt*k1,   **kw)
    k3 = state_derivative(t+0.5*dt,   state+0.5*dt*k2,   **kw)
    k4 = state_derivative(t+dt,       state+dt*k3,        **kw)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ────────────────────────────────────────────────
# GÉOMÉTRIE & RENDU DU BATEAU
# ────────────────────────────────────────────────
# Franc-bord / tirant d'eau : navire de charge typique, creux D ≈ 1.5·T,
# soit un franc-bord D − T ≈ 0.5·T (4 m pour T = 8 m).
DEFAULT_FREEBOARD_RATIO = 0.5

def create_boat_mesh(length=L, width=B, draft=T, freeboard=None,
                     Nu=25, Nv=9):
    """
    Coque FERMÉE : carène + pont plat. Le plan Z=0 (local) est la ligne de
    flottaison de conception : la quille (milieu) est à −draft, le livet
    (bord du pont) à +freeboard.

    - Vue en plan : étrave en pointe (x > 0), arrière légèrement affiné.
    - Sections pleines (profil en |v|⁴) : murailles quasi verticales, fond
      plat au milieu ; profil longitudinal en |u|⁴ : corps cylindrique au
      centre, fonds qui remontent vers l'étrave et l'arrière.

    Retourne (verts, tris, is_deck) — is_deck[i] vrai pour les triangles du
    pont (couleur propre dans Boat).
    """
    if freeboard is None:
        freeboard = DEFAULT_FREEBOARD_RATIO * draft
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
    depth = (1 - XI**4) * (1 - np.abs(ETA)**4)
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
    def __init__(self, length=L, width=B, draft=T, freeboard=None,
                 x0=0.0, y0=0.0, speed=1.0, Nu=25, Nv=9,
                 color='#FFD700', underwater_color='#2E8B57',
                 deck_color='#9C7A54'):
        if freeboard is None:
            freeboard = DEFAULT_FREEBOARD_RATIO * draft
        self.length=length; self.width=width
        self.draft=draft; self.freeboard=freeboard
        self.x0=x0; self.y0=y0; self.speed=speed
        self.local_vertices, self.triangles, self._is_deck = create_boat_mesh(
            length=length, width=width, draft=draft, freeboard=freeboard,
            Nu=Nu, Nv=Nv)

        def _valid(c, fallback):
            try:
                mcolors.to_rgb(c); return c
            except Exception:
                return fallback
        self.color = _valid(color, '#FFD700')
        self.underwater_color = _valid(underwater_color, '#2E8B57')
        self.deck_color = _valid(deck_color, '#9C7A54')
        # Couleur hors de l'eau : pont ou coque
        self._dry_colors = np.where(self._is_deck, self.deck_color, self.color)

        # Coloration de repli (ligne de flottaison de conception, Z local=0)
        # utilisée quand render() ne reçoit pas de fonction d'élévation de
        # mer (sea_elev_fn) — sinon l'immersion réelle est recalculée à
        # chaque frame dans render(), voir plus bas.
        # Un triangle n'est classé "immergé" que si ses 3 sommets sont sous
        # le seuil : un critère sur la moyenne des sommets laisse une
        # frontière ambiguë (mouchetée), où deux triangles voisins d'un même
        # carreau peuvent basculer chacun d'un côté différent.
        tri_local_z = self.local_vertices[self.triangles][:, :, 2]
        fully_submerged = np.all(tri_local_z < 0, axis=1)
        self._tri_colors = np.where(fully_submerged,
                                     self.underwater_color, self._dry_colors)

        self._plot=None; self._waterline=None

    def compute_mesh(self, pitch, roll, x_pos, y_pos, z_center, yaw=0.0,
                      sea_elev_fn=None):
        """
        Calcule (sans rien dessiner) les triangles de la coque en position
        monde, leur couleur, et le contour de la ligne de flottaison.

        sea_elev_fn(x_arr, y_arr) -> élévation locale de la mer (tableau),
        si fournie : la coloration immergé/émergé est recalculée à cette
        frame à partir de la vraie surface de mer sous chaque sommet de la
        coque (immersion dynamique), plutôt que d'utiliser le seuil de
        conception statique (self._tri_colors).

        Séparé de render() pour permettre de fusionner ces triangles avec
        ceux de la mer dans une seule Poly3DCollection côté appelant :
        mplot3d ne trie par profondeur qu'à l'intérieur d'une même
        collection, jamais entre deux artistes séparés — sans fusion, la
        mer ou le bateau serait entièrement devant/derrière l'autre au lieu
        d'une occlusion locale correcte.
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

        tol=max(0.05,0.02*(self.draft+self.freeboard))
        mask=np.abs(verts[:,2]-z_center)<tol; pts=verts[mask]
        waterline = None
        if pts.shape[0]>=3:
            c=pts[:,:2].mean(0)
            ang=np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0]); p=pts[np.argsort(ang)]
            xs=np.append(p[:,0],p[0,0]); ys=np.append(p[:,1],p[0,1]); zs=np.append(p[:,2],p[0,2])
            waterline = (xs, ys, zs)

        return tri_verts, tri_colors, waterline

    def render(self, ax, pitch, roll, x_pos, y_pos, z_center, yaw=0.0,
               sea_elev_fn=None):
        """
        Rendu autonome du bateau seul (sans fusion avec la mer) — pratique
        pour des tests ou un usage sans occlusion mer/bateau. Pour un rendu
        avec occlusion correcte, utiliser compute_mesh() et fusionner ses
        triangles avec ceux de la mer avant de créer la Poly3DCollection
        (voir main.py).
        """
        for attr in ('_plot','_waterline'):
            obj=getattr(self,attr)
            if obj is not None:
                try: obj.remove()
                except: pass
        tri_verts, tri_colors, waterline = self.compute_mesh(
            pitch, roll, x_pos, y_pos, z_center, yaw=yaw, sea_elev_fn=sea_elev_fn)

        self._plot=Poly3DCollection(tri_verts,
            facecolor=tri_colors, edgecolor='black',
            linewidth=1.2, alpha=1.0, antialiased=True, zorder=10)
        ax.add_collection3d(self._plot)
        if waterline is not None:
            xs, ys, zs = waterline
            self._waterline=ax.plot(xs,ys,zs,color='k',linewidth=2,zorder=11)[0]
        else:
            self._waterline=None