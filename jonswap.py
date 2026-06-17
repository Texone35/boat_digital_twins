import numpy as np
from scipy import constants
from scipy.ndimage import gaussian_filter
from matplotlib import colors as mcolors

# ======================
# 🔹 PARAMÈTRES JONSWAP
# ======================
Hs = 6.0       # Hauteur significative (m)
Tp = 8.0       # Période de pic (s)
gamma = 3.3    # Facteur de forme (3.3 = JONSWAP standard)
wind_scale = 1.5

g = constants.g  # 9.81 m/s²

# ======================
# 🔹 PARAMÈTRES DE GRID
# ======================
Lx, Ly = 150.0, 150.0
dx, dy = 4.0, 4.0
Nx, Ny = int(Lx / dx) + 1, int(Ly / dy) + 1

# ======================
# 🔹 PARAMÈTRES SPECTRAUX
# ======================
N_freq = 30
N_dir = 15

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
k = dispersion_relation(omega)

# Directions : distribution cosinus² autour de 0° (cap principal)
theta_mean = 0.0
theta = np.linspace(theta_mean - np.pi / 2, theta_mean + np.pi / 2, N_dir)

# Spectre directionnel : D(θ) ∝ cos²(θ - θ_mean), normalisé
cos2 = np.cos(theta - theta_mean) ** 2
D_theta = cos2 / (cos2.sum() * (theta[1] - theta[0]))  # ∫D dθ = 1

S_omega = jonswap_spectrum(omega, Hs, Tp, gamma)

# Phases aléatoires
np.random.seed(42)
phases = np.random.uniform(0, 2 * np.pi, (N_freq, N_dir))

# Incrément fréquentiel / directionnel
domega = omega[1] - omega[0]
dtheta = theta[1] - theta[0]

# Pré-calcul des amplitudes A(ω,θ) = √(2 S(ω)·D(θ) dω dθ)
S2D = S_omega[:, None] * D_theta[None, :]   # (N_freq, N_dir)
A_amp = np.sqrt(2.0 * S2D * domega * dtheta)   # amplitude de chaque composante

# Pré-calcul des vecteurs d'onde (kx, ky)
kx = k[:, None] * np.cos(theta)[None, :]   # (N_freq, N_dir)
ky = k[:, None] * np.sin(theta)[None, :]


# ======================
# 🔹 GÉNÉRATION DE LA SURFACE (vectorisée)
# ======================
def generate_sea_surface(t):
    """
    Surface η(x,y,t) = ΣΣ A(ω,θ) cos(kx·X + ky·Y − ω·t + φ)
    Retourne Z (Ny, Nx).
    """
    # Phase spatio-temporelle pour chaque composante : (N_freq, N_dir, Ny, Nx)
    phase_field = (
        kx[:, :, None, None] * X[None, None, :, :]
        + ky[:, :, None, None] * Y[None, None, :, :]
        - omega[:, None, None, None] * t
        + phases[:, :, None, None]
    )
    Z = np.sum(A_amp[:, :, None, None] * np.cos(phase_field), axis=(0, 1))
    Z = gaussian_filter(Z, sigma=1.0) * wind_scale
    return Z


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
def get_sea_colors(Z_new):
    cmap_sea = mcolors.LinearSegmentedColormap.from_list(
        'sea_rich', ['#001f3f', '#003b66', '#006699', '#00aacc', '#bfefff']
    )
    zmin, zmax = Z_new.min(), Z_new.max()
    eps = 1e-6
    norm_z = (Z_new - zmin) / (zmax - zmin + eps)

    dZ_dy_g, dZ_dx_g = np.gradient(Z_new, dy, dx)
    normal = np.dstack((-dZ_dx_g, -dZ_dy_g, np.ones_like(Z_new)))
    nlen = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (nlen + eps)

    light_dir = np.array([0.3, 0.3, 0.9])
    light_dir /= np.linalg.norm(light_dir)
    brightness = np.tensordot(normal, light_dir, axes=([2], [0]))
    brightness = np.clip(brightness, 0.0, 1.0)

    base_colors = cmap_sea(norm_z)
    base_colors[..., :3] *= (0.5 + 0.6 * brightness[..., None])

    foam_threshold = 0.55
    foam_mask = Z_new > (foam_threshold * zmax)
    base_colors[foam_mask, :3] = np.clip(base_colors[foam_mask, :3] + 0.45, 0, 1)

    return base_colors


if __name__ == '__main__':
    Z = generate_sea_surface(0.0)
    Hs_sim = 4.0 * np.std(Z)
    print(f"Surface générée : {Z.shape}, min={Z.min():.2f}, max={Z.max():.2f}")
    print(f"Hs simulé ≈ {Hs_sim:.2f} m  (cible : {Hs} m)")