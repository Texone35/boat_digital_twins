import numpy as np
from scipy import constants
from matplotlib import colors as mcolors

# ======================
# 🔹 PARAMÈTRES JONSWAP
# ======================
Hs = 16.0      # Hauteur significative (m) — ouragan (tempête : 12.0, mer agitée : 6.0)
Tp = 14.0      # Période de pic (s)              — ouragan (tempête : 11.0, agitée : 8.0)
gamma = 3.3    # Facteur de forme (3.3 = JONSWAP standard)
wind_scale = 1.0   # 1.0 : Hs de la mer simulée = Hs demandé

g = constants.g  # 9.81 m/s²

# ======================
# 🔹 PARAMÈTRES DE GRID
# ======================
Lx, Ly = 600.0, 600.0   # ≥ 2 longueurs d'onde au pic (λp = g·Tp²/2π ≈ 306 m)
dx, dy = 4.0, 4.0   # résout les vagues les plus courtes (λ = λp/16 ≈ 19 m à 4·ωp)
Nx, Ny = int(Lx / dx) + 1, int(Ly / dy) + 1

# ======================
# 🔹 PARAMÈTRES SPECTRAUX
# ======================
N_freq = 30
N_dir = 17

# Vagues « choppy » (Tessendorf 2001) : déplacement horizontal des points
# de la surface vers les crêtes → crêtes pointues, creux aplatis.
# Affichage uniquement : la physique utilise l'élévation eulérienne η(x,y).
CHOPPINESS = 0.8

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
# 🔹 PRÉ-CALCULS SPECTRAUX
# ======================
omega_min = 0.5 * (2 * np.pi / Tp)   # ½·ωp
omega_max = 4.0 * (2 * np.pi / Tp)   # 4·ωp  (couvre l'énergie utile)
omega = np.linspace(omega_min, omega_max, N_freq)
domega = omega[1] - omega[0]
# Fréquences tirées au hasard dans chaque bande dω : casse la périodicité
# (sinon la mer se répète exactement toutes les 2π/dω ≈ 66 s)
np.random.seed(42)
omega = omega + np.random.uniform(-0.5, 0.5, N_freq) * domega
k = dispersion_relation(omega)

# Directions : distribution cosinus² autour de 0° (cap principal)
theta_mean = 0.0
theta = np.linspace(theta_mean - np.pi / 2, theta_mean + np.pi / 2, N_dir)

# Spectre directionnel : D(θ) ∝ cos²(θ - θ_mean), normalisé
cos2 = np.cos(theta - theta_mean) ** 2
D_theta = cos2 / (cos2.sum() * (theta[1] - theta[0]))  # ∫D dθ = 1

S_omega = jonswap_spectrum(omega, Hs, Tp, gamma)

# Phases aléatoires
phases = np.random.uniform(0, 2 * np.pi, (N_freq, N_dir))

# Incrément directionnel
dtheta = theta[1] - theta[0]

# Pré-calcul des amplitudes A(ω,θ) = √(2 S(ω)·D(θ) dω dθ)
S2D = S_omega[:, None] * D_theta[None, :]   # (N_freq, N_dir)
A_amp = np.sqrt(2.0 * S2D * domega * dtheta)   # amplitude de chaque composante
# Normalisation exacte : Hs = 4·σ_η = 4·√(ΣA²/2). La formule de α ci-dessus
# ignore le gain d'énergie du pic JONSWAP (≈ ×1.5 pour γ = 3.3).
A_amp *= Hs / (4.0 * np.sqrt(np.sum(A_amp**2) / 2.0))

# Pré-calcul des vecteurs d'onde (kx, ky)
kx = k[:, None] * np.cos(theta)[None, :]   # (N_freq, N_dir)
ky = k[:, None] * np.sin(theta)[None, :]

# Facteurs séparables de la somme spectrale (voir generate_sea_surface) :
# e^{i(kx·x + ky·y)} = e^{i·ky·y} · e^{i·kx·x}, indépendants du temps donc
# précalculés une seule fois.  Composantes aplaties : C = N_freq·N_dir.
# complex64 : produits matriciels ≈2× plus rapides, erreur ~1e-6 m.
_Ey = np.exp(1j * np.outer(y, ky.ravel())).astype(np.complex64)   # (Ny, C)
_Ex = np.exp(1j * np.outer(x, kx.ravel())).astype(np.complex64)   # (Nx, C)
_c0 = (A_amp * np.exp(1j * phases)).ravel()          # (C,)
_omega_c = np.broadcast_to(omega[:, None], kx.shape).ravel()
# Coefficients spectraux des 3 champs (η, Dx, Dy) : Dx = −λ ΣA·(kx/k)·sin
# ⇔ Re[Σ (iλ·kx/k)·c·e^{i(...)}], idem Dy.
_k_c = np.hypot(kx, ky).ravel()
_field_coef = np.stack([np.ones_like(_k_c),
                        1j * CHOPPINESS * kx.ravel() / _k_c,
                        1j * CHOPPINESS * ky.ravel() / _k_c])   # (3, C)


# ======================
# 🔹 GÉNÉRATION DE LA SURFACE (vectorisée)
# ======================
def generate_sea_surface(t):
    """
    Surface η(x,y,t) = ΣΣ A(ω,θ) cos(kx·X + ky·Y − ω·t + φ)
    Retourne Z (Ny, Nx).

    Écrite comme Re[ Ey · diag(A·e^{i(φ−ωt)}) · Exᵀ ] : un seul produit
    matriciel (Ny×C)(C×Nx) au lieu d'évaluer C·Ny·Nx cosinus par frame.
    """
    c_t = (_c0 * np.exp(-1j * _omega_c * t)).astype(np.complex64)
    return ((_Ey * c_t) @ _Ex.T).real.astype(float) * wind_scale


def generate_sea_fields(t):
    """
    Champs de mer au temps t pour l'affichage : (Z, Dx, Dy), chacun (Ny, Nx).
    Z = élévation eulérienne (identique à generate_sea_surface, utilisée par
    la physique) ; Dx, Dy = déplacements horizontaux « choppy » des sommets.
    """
    c_t = (_field_coef * (_c0 * np.exp(-1j * _omega_c * t))).astype(np.complex64)
    F = ((_Ey[None] * c_t[:, None, :]) @ _Ex.T).real.astype(float) * wind_scale
    return F[0], F[1], F[2]


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


def sea_gradient_at(Z, xb, yb):
    """
    Gradient local de la surface ∂η/∂x, ∂η/∂y à la position du bateau.
    Utilisé pour calculer les angles de houle induits.
    """
    dZ_dy_grid, dZ_dx_grid = np.gradient(Z, dy, dx)
    dzdx = interpolate_sea_at(dZ_dx_grid, xb, yb)
    dzdy = interpolate_sea_at(dZ_dy_grid, xb, yb)
    return dzdx, dzdy


# ======================
# 🔹 COULEURS DE LA MER
# ======================
_SEA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'sea_deep', ['#021526', '#063a5c', '#0f5f7d', '#1f8a9a', '#5cc2c4'])
_FOAM_RGB = np.array([0.93, 0.96, 0.98])


def _smoothstep(e0, e1, v):
    u = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def get_sea_colors_and_normals(Z, Dx, Dy):
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

    h = np.clip(0.5 + Z / (1.2 * Hs * wind_scale), 0.0, 1.0)
    rgb = _SEA_CMAP(h)[..., :3]
    J = (1.0 + dDx_dx) * (1.0 + dDy_dy) - dDx_dy * dDy_dx
    foam = _smoothstep(0.75, 0.35, J) * _smoothstep(0.0, 0.3 * Hs * wind_scale, Z)
    rgb += (_FOAM_RGB - rgb) * (0.85 * foam[..., None])

    n = np.cross(np.stack([1.0 + dDx_dx, dDy_dx, Zx], axis=-1),
                 np.stack([dDx_dy, 1.0 + dDy_dy, Zy], axis=-1)).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return rgb, n


def get_sea_colors(Z, Dx, Dy):
    """Couleur RGB (Ny, Nx, 3) seule — voir get_sea_colors_and_normals."""
    return get_sea_colors_and_normals(Z, Dx, Dy)[0]


if __name__ == '__main__':
    Z = generate_sea_surface(0.0)
    Hs_sim = 4.0 * np.std(Z)
    print(f"Surface générée : {Z.shape}, min={Z.min():.2f}, max={Z.max():.2f}")
    print(f"Hs simulé ≈ {Hs_sim:.2f} m  (cible : {Hs * wind_scale} m)")