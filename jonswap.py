import numpy as np
from scipy import constants
from matplotlib import colors as mcolors

g = constants.g  # 9.81 m/s²

# ======================
# 🔹 PARAMÈTRES DE GRID (fixes : la grille ne change jamais)
# ======================
Lx, Ly = 600.0, 600.0   # ≥ 2 longueurs d'onde au pic (λp = g·Tp²/2π ≈ 306 m à Tp = 14 s)
dx, dy = 4.0, 4.0   # résout les vagues les plus courtes (λ = λp/16 ≈ 19 m à 4·ωp)
Nx, Ny = int(Lx / dx) + 1, int(Ly / dy) + 1

# ======================
# 🔹 PARAMÈTRES SPECTRAUX
# ======================
N_freq = 30
N_dir = 17
# Composantes plus courtes que 3 mailles : mal résolues par la grille
# (repliement de spectre aux petits Tp) → amplitude mise à zéro.
K_MAX = 2 * np.pi / (3 * dx)

# ======================
# 🔹 ÉTAT DE MER PAR DÉFAUT ET PRÉRÉGLAGES
# ======================
# Hs : hauteur significative (m) ; Tp : période de pic (s) ; gamma : facteur
# de forme (3.3 = JONSWAP standard) ; theta_mean : direction de propagation
# (rad, 0 = vers +x) ; spread : exposant s de D(θ) ∝ cos^{2s}(θ − θ_mean)
# (grand = houle plus unidirectionnelle) ; choppiness : vagues « choppy »
# (Tessendorf 2001), affichage uniquement — la physique utilise η(x,y).
DEFAULTS = dict(Hs=16.0, Tp=14.0, gamma=3.3, theta_mean=0.0,
                spread=1.0, choppiness=0.8)

SEA_PRESETS = {
    'Calme':   dict(Hs=1.0,  Tp=6.0),
    'Agitée':  dict(Hs=6.0,  Tp=8.0),
    'Tempête': dict(Hs=12.0, Tp=11.0),
    'Ouragan': dict(Hs=16.0, Tp=14.0),
}

# Constante de temps des transitions d'état de mer (s) : un changement de
# paramètre est atteint à 63 % après TAU, sans saut de la surface.
TAU = 3.0
# Écart en dessous duquel un paramètre rejoint directement sa cible
_SNAP = dict(Hs=0.01, Tp=0.01, gamma=0.01, theta_mean=np.radians(0.2),
             spread=0.01, choppiness=0.005)
# Paramètres qui changent ω ou k : facteurs spatiaux à recalculer
_GEOMETRIC = ('Tp', 'theta_mean')

# ======================
# 🔹 GRILLE SPATIALE
# ======================
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# ======================
# 🔹 SPECTRE JONSWAP CORRIGÉ
# ======================
def jonswap_spectrum(omega, Hs, Tp, gamma):
    """
    Spectre JONSWAP normalisé correctement.
    Vérifie : ∫S(ω)dω = (Hs/4)²  (variance spectrale)
    """
    omega_p = 2 * np.pi / Tp
    # Spectre de Pierson-Moskowitz
    alpha = (Hs ** 2) * (omega_p ** 4) / (4.0 * g ** 2)  # coefficient de Phillips adapté
    S_pm = alpha * g**2 / omega**5 * np.exp(-5.0 / 4.0 * (omega_p / omega) ** 4)
    # Pic JONSWAP
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    r = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    S_jonswap = S_pm * (gamma ** r)
    S_jonswap = np.where(omega > 0, S_jonswap, 0.0)
    return S_jonswap

# ======================
# 🔹 RELATION DE DISPERSION (eaux profondes)
# ======================
def dispersion_relation(omega, h=np.inf):
    """k = ω²/g  (eaux profondes, h→∞)"""
    if np.isinf(h):
        return omega ** 2 / g
    else:
        # eaux finies : résolution itérative de ω² = g·k·tanh(k·h)
        k0 = omega ** 2 / g
        for _ in range(50):
            k0 = omega ** 2 / (g * np.tanh(k0 * h))
        return k0


# ======================
# 🔹 COMPOSANTES SPECTRALES (partagées avec wavepinn)
# ======================
def spectral_components(Hs, Tp, gamma, theta_mean, spread, jitter, n_dir=N_dir,
                        w_band=(0.5, 4.0), k_max=K_MAX):
    """
    Composantes de la mer JONSWAP directionnelle : une seule source pour la
    mer affichée (SeaState) et le générateur de vérité (wavepinn).

    len(jitter) fixe le nombre de fréquences ; chaque fréquence est décalée
    de jitter·dω dans sa bande (casse la périodicité). n_dir = 1 : houle
    unidirectionnelle vers theta_mean. Amplitudes A = √(2 S(ω) D(θ)dθ dω),
    renormalisées pour que Hs = 4·√(ΣA²/2) exactement.

    Retourne (ω (Nf,), dω, k (Nf,), θ (Nd,), A (Nf, Nd)).
    """
    wp = 2 * np.pi / Tp
    w = np.linspace(w_band[0] * wp, w_band[1] * wp, len(jitter))
    dw = w[1] - w[0]
    w = w + np.asarray(jitter) * dw
    offsets = np.linspace(-np.pi / 2, np.pi / 2, n_dir) if n_dir > 1 else np.zeros(1)
    theta = theta_mean + offsets
    k = dispersion_relation(w)
    S = jonswap_spectrum(w, Hs, Tp, gamma)
    # D(θ)dθ ∝ cos^{2s}, somme 1 (cos ≥ 0 sur ±π/2 ; clip : arrondis aux bords)
    c = np.clip(np.cos(offsets), 0.0, None) ** (2 * spread)
    A = np.sqrt(2.0 * S[:, None] * (c / c.sum())[None, :] * dw)
    A[k > k_max, :] = 0.0
    # Normalisation exacte : la formule de α ignore le gain d'énergie du pic
    # JONSWAP (≈ ×1.5 pour γ = 3.3).
    A *= Hs / (4.0 * np.sqrt(np.sum(A**2) / 2.0))
    return w, dw, k, theta, A


def _sea_args(p):
    return p['Hs'], p['Tp'], p['gamma'], p['theta_mean'], p['spread']


# ======================
# 🔹 ÉTAT DE MER RECONFIGURABLE EN DIRECT
# ======================
class SeaState:
    """
    Mer JONSWAP directionnelle dont les paramètres se modifient en cours
    de simulation.

    Surface η(x,y,t) = ΣΣ A(ω,θ) cos(kx·x + ky·y − ω·t + φ), écrite comme
    Re[ Ey · diag(A·e^{i(φ−ωt)}) · Exᵀ ] : un seul produit matriciel
    (Ny×C)(C×Nx) au lieu d'évaluer C·Ny·Nx cosinus par frame.

    Les tirages aléatoires (décalages de fréquence, phases) sont faits une
    seule fois : la mer garde sa « personnalité » quand ses paramètres
    changent. set_target() fixe une cible, advance() s'en rapproche
    progressivement (constante de temps TAU).
    """

    def __init__(self, seed=42, **params):
        p = dict(DEFAULTS)
        p.update(params)
        self.params = p
        self.target = dict(p)

        rng = np.random.RandomState(seed)
        # Fréquences tirées au hasard dans chaque bande dω : casse la
        # périodicité (sinon la mer se répète toutes les 2π/dω ≈ 66 s)
        self._jitter = rng.uniform(-0.5, 0.5, N_freq)
        self._phi = rng.uniform(0, 2 * np.pi, N_freq * N_dir)

        # Instant et point (en général le bateau) où la surface est gardée
        # continue lors d'un changement de ω ou de k
        self._t = 0.0
        self._ref = (Lx / 2, Ly / 2)
        self._omega_c = None
        self._build_geometry()
        self._build_amplitudes()

    # ── Accès aux paramètres courants ──
    Hs = property(lambda self: self.params['Hs'])
    Tp = property(lambda self: self.params['Tp'])
    gamma = property(lambda self: self.params['gamma'])
    theta_mean = property(lambda self: self.params['theta_mean'])
    omega_p = property(lambda self: 2 * np.pi / self.params['Tp'])

    def _build_geometry(self):
        """Fréquences, directions, vecteurs d'onde et facteurs e^{i·k·x}."""
        w, dw, k, theta, _ = spectral_components(*_sea_args(self.params), self._jitter)
        kx =(k[:, None] * np.cos(theta)[None, :]).ravel()   # composantes aplaties
        ky = (k[:, None] * np.sin(theta)[None, :]).ravel()   # C = N_freq·N_dir
        omega_c = np.repeat(w, N_dir)

        if self._omega_c is not None:
            # Continuité de phase : chaque composante garde la même phase à
            # l'instant courant et au point de référence,
            # φ' = φ + (ω' − ω)·t − (k' − k)·x_ref
            xr, yr = self._ref
            self._phi = self._phi + ((omega_c - self._omega_c) * self._t
                                     - (kx - self._kx) * xr
                                     - (ky - self._ky) * yr)

        self.omega, self.domega, self.k = w, dw, k
        self.theta, self.dtheta = theta, theta[1] - theta[0]
        self._omega_c, self._kx, self._ky = omega_c, kx, ky
        self._k_c = np.hypot(kx, ky)
        # Facteurs séparables, indépendants du temps. complex64 : produits
        # matriciels ≈2× plus rapides, erreur ~1e-6 m.
        self._Ey = np.exp(1j * np.outer(y, ky)).astype(np.complex64)   # (Ny, C)
        self._Ex = np.exp(1j * np.outer(x, kx)).astype(np.complex64)   # (Nx, C)

    def _build_amplitudes(self):
        """Amplitudes A(ω,θ) (spectral_components) et coefficients choppy."""
        p = self.params
        self._A = spectral_components(*_sea_args(p), self._jitter)[4].ravel()
        # Coefficients spectraux des 3 champs (η, Dx, Dy) : Dx = −λ ΣA·(kx/k)·sin
        # ⇔ Re[Σ (iλ·kx/k)·c·e^{i(...)}], idem Dy.
        lam = p['choppiness']
        self._field_coef = np.stack([np.ones_like(self._k_c),
                                     1j * lam * self._kx / self._k_c,
                                     1j * lam * self._ky / self._k_c])   # (3, C)

    # ── Pilotage en direct ──
    def set_target(self, **params):
        """Nouvelle cible pour un ou plusieurs paramètres (transition douce)."""
        for key, v in params.items():
            if key not in self.target:
                raise KeyError(f'paramètre de mer inconnu : {key}')
            self.target[key] = float(v)

    def set_params(self, **params):
        """Changement immédiat (sans transition)."""
        self.set_target(**params)
        self.advance(np.inf)

    def advance(self, dt, ref=None):
        """
        Rapproche les paramètres courants de la cible sur une durée dt (s).
        ref = (x, y) : point où la surface reste continue (le bateau).
        Retourne True si la mer a changé.
        """
        if ref is not None:
            self._ref = ref
        a = 1.0 - np.exp(-dt / TAU)
        geom = changed = False
        for key, q in self.target.items():
            p = self.params[key]
            d = q - p
            if key == 'theta_mean':
                d = (d + np.pi) % (2 * np.pi) - np.pi        # plus court chemin
            if d == 0.0:
                continue
            self.params[key] = p + (d if abs(d) < _SNAP[key] else a * d)
            changed = True
            geom |= key in _GEOMETRIC
        if geom:
            self._build_geometry()
        if changed:
            self._build_amplitudes()
        return changed

    def is_transitioning(self):
        return any(self.params[k] != v for k, v in self.target.items())

    # ── Génération de la surface ──
    def generate_fields(self, t):
        """
        Champs de mer au temps t : (Z, Dx, Dy), chacun (Ny, Nx).
        Z = élévation eulérienne (utilisée par la physique) ;
        Dx, Dy = déplacements horizontaux « choppy » des sommets (affichage).
        """
        self._t = t
        c_t = (self._field_coef
               * (self._A * np.exp(1j * (self._phi - self._omega_c * t)))).astype(np.complex64)
        F = ((self._Ey[None] * c_t[:, None, :]) @ self._Ex.T).real.astype(float)
        return F[0], F[1], F[2]

    def generate_surface(self, t):
        """Élévation seule Z (Ny, Nx)."""
        return self.generate_fields(t)[0]

    def colors_and_normals(self, Z, Dx, Dy):
        """Voir get_sea_colors_and_normals ; échelle de teinte liée au Hs courant."""
        return get_sea_colors_and_normals(Z, Dx, Dy, self.Hs)


def interpolate_sea_at(Z, xb, yb):
    """
    Interpole bilinéairement l'élévation de la mer à la position (xb, yb) du bateau.
    """
    xi = np.clip(xb / dx, 0, Nx - 1)
    yi = np.clip(yb / dy, 0, Ny - 1)
    i0, j0 = int(xi), int(yi)
    i1, j1 = min(i0 + 1, Nx - 1), min(j0 + 1, Ny - 1)
    fx, fy = xi - i0, yi - j0
    z = (Z[j0, i0] * (1 - fx) * (1 - fy)
         + Z[j0, i1] * fx * (1 - fy)
         + Z[j1, i0] * (1 - fx) * fy
         + Z[j1, i1] * fx * fy)
    return z


def interpolate_sea_at_batch(Z, xb, yb):
    """
    Version vectorisée de interpolate_sea_at pour un tableau de points
    (xb, yb) — utilisée pour tester l'immersion de chaque sommet de la
    coque du bateau à chaque frame (coloration dynamique de la carène).
    """
    xb = np.asarray(xb, dtype=float)
    yb = np.asarray(yb, dtype=float)
    xi = np.clip(xb / dx, 0, Nx - 1)
    yi = np.clip(yb / dy, 0, Ny - 1)
    i0 = xi.astype(int); j0 = yi.astype(int)
    i1 = np.minimum(i0 + 1, Nx - 1); j1 = np.minimum(j0 + 1, Ny - 1)
    fx, fy = xi - i0, yi - j0
    z = (Z[j0, i0] * (1 - fx) * (1 - fy)
         + Z[j0, i1] * fx * (1 - fy)
         + Z[j1, i0] * (1 - fx) * fy
         + Z[j1, i1] * fx * fy)
    return z


# ======================
# 🔹 COULEURS DE LA MER
# ======================
_SEA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'sea_deep', ['#021526', '#063a5c', '#0f5f7d', '#1f8a9a', '#5cc2c4'])
_FOAM_RGB = np.array([0.93, 0.96, 0.98])


def _smoothstep(e0, e1, v):
    u = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def color_range(Hs):
    """Demi-plage d'élévation (m) couverte par l'échelle de teinte."""
    return 0.6 * Hs


def get_sea_colors_and_normals(Z, Dx, Dy, Hs):
    """
    Couleur RGB (Ny, Nx, 3) et normale unitaire (Ny·Nx, 3) de chaque sommet
    de la surface affichée P(x,y) = (x + Dx, y + Dy, Z). Les gradients sont
    calculés une seule fois pour les deux. L'éclairage (reflets, ombrage)
    est laissé au moteur de rendu : seule la teinte est fixée ici.

    - Teinte : bleu profond dans les creux → turquoise sur les crêtes, sur une
      échelle fixe liée à Hs (pas de min/max par frame → pas de scintillement).
    - Écume : là où la surface « choppy » se comprime (jacobien J du
      déplacement horizontal petit, Tessendorf 2001), c.-à-d. sur les crêtes
      raides — transition progressive, pas de seuil net.
    - Normale : n = ∂P/∂x × ∂P/∂y (différences finies).
    """
    Zy, Zx = np.gradient(Z, dy, dx)
    dDx_dy, dDx_dx = np.gradient(Dx, dy, dx)
    dDy_dy, dDy_dx = np.gradient(Dy, dy, dx)

    h = np.clip(0.5 + Z / (2.0 * color_range(Hs)), 0.0, 1.0)
    rgb = _SEA_CMAP(h)[..., :3]
    J = (1.0 + dDx_dx) * (1.0 + dDy_dy) - dDx_dy * dDy_dx
    foam = _smoothstep(0.75, 0.35, J) * _smoothstep(0.0, 0.3 * Hs, Z)
    rgb += (_FOAM_RGB - rgb) * (0.85 * foam[..., None])

    n = np.cross(np.stack([1.0 + dDx_dx, dDy_dx, Zx], axis=-1),
                 np.stack([dDx_dy, 1.0 + dDy_dy, Zy], axis=-1)).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return rgb, n


if __name__ == '__main__':
    sea = SeaState()
    for Hs, Tp in [(1.0, 6.0), (6.0, 8.0), (16.0, 14.0), (3.0, 4.0)]:
        sea.set_params(Hs=Hs, Tp=Tp)
        Z = sea.generate_surface(0.0)
        print(f"Hs={Hs:5.1f} m  Tp={Tp:4.1f} s  →  Hs simulé ≈ {4.0 * np.std(Z):5.2f} m"
              f"  (min={Z.min():+.2f}, max={Z.max():+.2f})")

    # Continuité : un changement de direction ne fait pas sauter la surface
    # au point de référence (seule la forme spatiale tourne autour de lui)
    sea.set_params(Hs=6.0, Tp=8.0, theta_mean=0.0)
    ref = (300.0, 300.0)
    z0 = interpolate_sea_at(sea.generate_surface(37.0), *ref)
    sea.advance(0.0, ref=ref)
    sea.set_params(theta_mean=np.radians(60))
    z1 = interpolate_sea_at(sea.generate_surface(37.0), *ref)
    print(f"Continuité au point de référence : {z0:+.3f} m → {z1:+.3f} m")
