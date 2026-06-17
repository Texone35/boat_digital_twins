"""
boat.py — Modèle hydrodynamique 6 DDL basé sur les équations de Cummins (1962).

Équation de Cummins linéarisée (Ogilvie 1964) :
    [M + A(ω_e)] η̈ + B(ω_e) η̇ + C η = F_exc(ω_e, t)

La fréquence de rencontre ω_e est calculée à chaque pas de temps
depuis la vitesse du navire et l'angle de rencontre avec la houle.
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import colors as mcolors

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
# PARAMÈTRES DE HOULE
# ────────────────────────────────────────────────
Hs           = 3.0
Tp           = 8.0
omega_p      = 2 * np.pi / Tp
wave_heading = np.pi / 4   # 45° : excite roll et pitch simultanément
A_wave       = Hs / 2.0

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
# RAO(ω_e) — modèle oscillateur amorti
# ────────────────────────────────────────────────
_RAO_ref  = np.array([0.40, 0.25, 0.60, 0.08, 0.10, 0.04])
_ZETA_RAO = np.array([0.30, 0.30, 0.10, 0.12, 0.08, 0.40])

def rao_at(omega_e, dof):
    """
    RAO(ω_e) = RAO_ref · H(ω_e) / H(ω_p)
    H(ω) = 1 / √[(1−r²)² + (2ζr)²]  avec r = ω_e/ω_n
    """
    wn  = _omega_n_all[dof]
    zeta = _ZETA_RAO[dof]
    def H(w):
        r = w / wn
        return 1.0 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
    return _RAO_ref[dof] * H(omega_e) / H(omega_p)

# ────────────────────────────────────────────────
# FORCES D'EXCITATION DE HOULE — Froude-Krylov + diffraction
# ────────────────────────────────────────────────
_rng    = np.random.RandomState(42)
_phase6 = _rng.uniform(0, 2*np.pi, 6)

def excitation_forces(t, omega_e, mu, sea_elev, sea_dzdx, sea_dzdy):
    """
    F_exc_i = X_i(ω_e) · a_eff · cos(ω_e·t + φ_i)

    X_i = RAO_i(ω_e) × C_ii_ref  (coefficient d'excitation dimensionnel)

    Les pentes de la mer (sea_dzdx, sea_dzdy) ajoutent un moment
    de Froude-Krylov directionnel pour roll et pitch.

    IMPORTANT : on n'ajoute pas C_ii·sea_elev ici car le terme
    de restauration −C·η inclut déjà la flottaison hydrostatique.
    La surface libre effective est capturée via a_eff.
    """
    # Amplitude locale effective — bornée à 2·A_wave
    a_eff = float(np.clip(abs(sea_elev), 1e-3, 2.0 * A_wave))
    # Signe de l'élévation (phase)
    sgn   = np.sign(sea_elev) if abs(sea_elev) > 1e-4 else 1.0

    # Pentes locales bornées
    sx = float(np.clip(sea_dzdx, -0.25, 0.25))
    sy = float(np.clip(sea_dzdy, -0.25, 0.25))

    F = np.zeros(6)

    # ── Surge (0) — excitation longitudinale faible ──
    F[0] = (rao_at(omega_e,0) * rho*g*Awp * 0.01
            * a_eff * np.cos(mu) * np.cos(omega_e*t + _phase6[0]))

    # ── Sway (1) — excitation transversale ──
    F[1] = (rao_at(omega_e,1) * rho*g*Awp * 0.01
            * a_eff * np.sin(mu) * np.cos(omega_e*t + _phase6[1]))

    # ── Heave (2) — pression de houle sur la flottaison ──
    # F_FK = ρg·Awp · a_eff · RAO_heave(ω_e) · cos(ω_e·t)
    F[2] = (rao_at(omega_e,2) * rho*g*Awp
            * a_eff * np.cos(omega_e*t + _phase6[2]))

    # ── Roll (3) — moment dû à l'inclinaison transversale de la houle ──
    # Composante harmonique (RAO)
    M3_harm  = (rao_at(omega_e,3) * C_mat[3,3]
                * a_eff * abs(np.sin(mu)) * np.cos(omega_e*t + _phase6[3]))
    # Composante statique (pente réelle de la mer sous la coque)
    # M_roll_slope = ρg·∇·(∂η/∂y)·sin(μ)  — moment de Froude-Krylov local
    M3_slope = C_mat[3,3] * sy * abs(np.sin(mu))
    F[3] = M3_harm + M3_slope

    # ── Pitch (4) — moment dû à l'inclinaison longitudinale ──
    M4_harm  = (rao_at(omega_e,4) * C_mat[4,4]
                * a_eff * abs(np.cos(mu)) * np.cos(omega_e*t + _phase6[4]))
    M4_slope = C_mat[4,4] * sx * abs(np.cos(mu))
    F[4] = M4_harm + M4_slope

    # ── Yaw (5) ──
    F[5] = (rao_at(omega_e,5) * C_mat[4,4] * 0.1
            * a_eff * np.sin(mu)*np.cos(mu) * np.cos(omega_e*t + _phase6[5]))

    return F

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
def state_derivative(t, state, sea_elev=0.0, sea_dzdx=0.0, sea_dzdy=0.0):
    """
    [M + A(ω_e)] η̈ + B(ω_e) η̇ + C η = F_exc(ω_e, t)

    state = [η(6), η̇(6)]
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
           + excitation_forces(t, omega_e, mu, sea_elev, sea_dzdx, sea_dzdy))

    nu_dot = np.linalg.solve(MA, rhs)
    return np.concatenate([nu, nu_dot])

# ────────────────────────────────────────────────
# INTÉGRATION RK4
# ────────────────────────────────────────────────
def step_state(state, t, dt, sea_elev=0.0, sea_dzdx=0.0, sea_dzdy=0.0):
    kw = dict(sea_elev=sea_elev, sea_dzdx=sea_dzdx, sea_dzdy=sea_dzdy)
    k1 = state_derivative(t,          state,              **kw)
    k2 = state_derivative(t+0.5*dt,   state+0.5*dt*k1,   **kw)
    k3 = state_derivative(t+0.5*dt,   state+0.5*dt*k2,   **kw)
    k4 = state_derivative(t+dt,       state+dt*k3,        **kw)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ────────────────────────────────────────────────
# GÉOMÉTRIE & RENDU DU BATEAU
# ────────────────────────────────────────────────
def create_boat_mesh(length=20.0, width=5.0, height=3.0,
                     Nu=25, Nv=9, color='#FFD700'):
    u = np.linspace(-length/2, length/2, Nu)
    v = np.linspace(-width/2,  width/2,  Nv)
    U, V = np.meshgrid(u, v, indexing='ij')
    lf  = np.clip(1-(2*U/length)**2, 0, 1)
    tf  = np.clip(1-(2*np.abs(V)/width)**2, 0, 1)
    Z   = (height - 1.8*lf*tf) - (height - 1.8*0.5)
    verts = np.vstack([U.flatten(), V.flatten(), Z.flatten()]).T
    tris  = []
    for i in range(Nu-1):
        for j in range(Nv-1):
            idx = i*Nv+j
            tris += [[idx,idx+1,idx+Nv],[idx+1,idx+Nv+1,idx+Nv]]
    try: mcolors.to_rgb(color)
    except Exception: color = '#8B4513'
    return verts, np.array(tris), color


class Boat:
    def __init__(self, length=20.0, width=5.0, height=3.0,
                 x0=0.0, y0=0.0, speed=1.0, Nu=25, Nv=9, color='#8B4513'):
        self.length=length; self.width=width; self.height=height
        self.x0=x0; self.y0=y0; self.speed=speed
        self.local_vertices, self.triangles, self.color = create_boat_mesh(
            length=length, width=width, height=height, Nu=Nu, Nv=Nv, color=color)
        self._plot=None; self._waterline=None

    def render(self, ax, pitch, roll, x_pos, y_pos, z_center, yaw=0.0):
        for attr in ('_plot','_waterline'):
            obj=getattr(self,attr)
            if obj is not None:
                try: obj.remove()
                except: pass
        Rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        Ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        verts=(Rz@Ry@Rx@self.local_vertices.T).T+np.array([x_pos,y_pos,z_center])
        self._plot=ax.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],
            triangles=self.triangles,color=self.color,
            alpha=1.0,linewidth=1.2,edgecolor='black',antialiased=True,zorder=10)
        tol=max(0.05,0.02*self.height)
        mask=np.abs(verts[:,2]-z_center)<tol; pts=verts[mask]
        if pts.shape[0]>=3:
            c=pts[:,:2].mean(0)
            ang=np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0]); p=pts[np.argsort(ang)]
            xs=np.append(p[:,0],p[0,0]); ys=np.append(p[:,1],p[0,1]); zs=np.append(p[:,2],p[0,2])
            self._waterline=ax.plot(xs,ys,zs,color='k',linewidth=2)[0]
        else: self._waterline=None