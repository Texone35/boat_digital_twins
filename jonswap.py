import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy import constants
from scipy.ndimage import gaussian_filter
from matplotlib import cm, colors as mcolors
from boat import Boat

# ======================
# 🔹 PARAMÈTRES JONSWAP (à ajuster)
# ======================
Hs = 4.0       # Hauteur significative (m)
Tp = 8.0       # Période de pic (s)
gamma = 3.3    # Facteur de forme (3.3 = JONSWAP standard)
wind_scale = 1.5  # Facteur de vent / agitation

g = constants.g  # 9.81 m/s²

# ======================
# 🔹 PARAMÈTRES DE GRID (optimisés pour fluidité)
# ======================
Lx, Ly = 100.0, 100.0  # Taille du domaine (m)
dx, dy = 4.0, 4.0      # Résolution spatiale (m)
Nx, Ny = int(Lx / dx) + 1, int(Ly / dy) + 1  # Nombre de points

# ======================
# 🔹 PARAMÈTRES SPECTRAUX
# ======================
N_freq = 30   # Nombre de fréquences
N_dir = 15    # Nombre de directions

# ======================
# 🔹 INITIALISATION DE LA GRILLE SPATIALE
# ======================
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# ======================
# 🔹 FONCTIONS : SPECTRE JONSWAP ET RELATION DE DISPERSION
# ======================
def jonswap_spectrum(omega, Hs, Tp, gamma):
    """Calcule le spectre JONSWAP."""
    omega_p = 2 * np.pi / Tp
    S_pm = (Hs ** 2) / (16 * Tp) * (omega_p ** 4) / (omega ** 5) * np.exp(-5 / 4 * (omega_p / omega) ** 4)
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    exponent = -0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2
    gamma_factor = gamma ** np.exp(exponent)
    S_jonswap = S_pm * gamma_factor
    S_jonswap[omega == 0] = 0
    return S_jonswap

def dispersion_relation(omega, h=np.inf):
    """Relation de dispersion pour l'eau profonde."""
    return omega ** 2 / g

# ======================
# 🔹 PRÉ-CALCULS
# ======================
omega_min = 0.1
omega_max = 3 * (2 * np.pi / Tp)
omega = np.linspace(omega_min, omega_max, N_freq)
k = dispersion_relation(omega)
theta = np.linspace(0, 2 * np.pi, N_dir, endpoint=False)
S_omega = jonswap_spectrum(omega, Hs, Tp, gamma)

# Phases aléatoires (fixes pour reproductibilité)
np.random.seed(42)
phases = np.random.uniform(0, 2 * np.pi, (N_freq, N_dir))

# ======================
# 🔹 INITIALISATION DE LA FIGURE 3D
# ======================
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(-Hs * 1.5, Hs * 1.5)
ax.set_title(f'Surface de mer JONSWAP + Navire (Hs={Hs}m, Tp={Tp}s, γ={gamma})', fontsize=14, pad=20)

# ======================
# 🔹 BATEAU 3D (séparé dans boat.py)
# ======================
# Position et dimensions (paramètres utilisateurs)
x_boat, y_boat = Lx / 2, Ly / 2
length, width, height = 20.0, 5.0, 3.0  # Dimensions (m)

# Crée une instance Boat (gère géométrie, couleur, position, rendu)
boat = Boat(length=length, width=width, height=height, x0=x_boat, y0=y_boat,
            speed=2.5, Nu=25, Nv=9, color='#8B4513')

# Parameters de simulation temporelle
dt_frame = 0.05  # time per frame (s)

# ======================
# 🔹 VARIABLE GLOBALE POUR LA SURFACE
# ======================
current_surf = None

# ======================
# 🔹 FONCTION DE MISE À JOUR (ANIMATION)
# ======================
def update(frame):
    """
    Fonction de mise à jour pour l'animation.
    Recalcule la surface de mer à chaque frame en fonction du temps t,
    et met à jour l'affichage 3D.

    Args:
        frame (int): Numéro de la frame (utilisé pour calculer le temps t).

    Returns:
        tuple: La surface mise à jour (pour FuncAnimation).
    """
    global current_surf  # Variable globale pour stocker la surface actuelle

    # ======================
    # 1️⃣ Calcul du temps actuel (t)
    # ======================
    t = frame * dt_frame

    # ======================
    # 2️⃣ Recalcul de l'élévation de la surface Z_new(t)
    # ======================
    # Initialisation de la matrice d'élévation (tous les points à 0)
    Z_new = np.zeros((Ny, Nx))

    # Superposition des ondes (méthode des phases aléatoires)
    for i in range(N_freq):
        for j in range(N_dir):
            # Composantes du nombre d'onde
            dk = k[i] * np.cos(theta[j])  # Composante x
            dl = k[i] * np.sin(theta[j])  # Composante y

            # Incréments pour l'intégration spectrale
            domega = omega[1] - omega[0] if N_freq > 1 else omega_max - omega_min
            dtheta = theta[1] - theta[0] if N_dir > 1 else 2 * np.pi

            # Amplitude de l'onde (liée au spectre JONSWAP)
            A = np.sqrt(2 * S_omega[i] * domega * dtheta)

            # Superposition avec déphasage temporel (-omega[i] * t)
            # C'est CE TERME qui fait évoluer les vagues dans le temps !
            Z_new += A * np.cos(
                dk * X + dl * Y - omega[i] * t + phases[i, j]
            )

    # ======================
    # 3️⃣ Mise à jour de la surface 3D
    # ======================
    # Supprimer l'ancienne surface si elle existe
    if current_surf is not None:
        current_surf.remove()

    # Lissage léger et atténuation pour rendre la mer plus "légère"
    Z_new = gaussian_filter(Z_new, sigma=1.0)
    Z_new *= 0.6 * wind_scale

    # Créer un colormap plus réaliste et calculer des couleurs de face
    cmap_sea = mcolors.LinearSegmentedColormap.from_list(
        'sea_rich', ['#001f3f', '#003b66', '#006699', '#00aacc', '#bfefff']
    )

    # Normaliser la hauteur pour la cartographie des couleurs
    zmin, zmax = Z_new.min(), Z_new.max()
    eps = 1e-6
    norm_z = (Z_new - zmin) / (zmax - zmin + eps)

    # Calculer normales approximatives pour un éclairage simple
    dZ_dy, dZ_dx = np.gradient(Z_new, dy, dx)
    normal = np.dstack((-dZ_dx, -dZ_dy, np.ones_like(Z_new)))
    nlen = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= (nlen + eps)

    # Simple éclairage directionnel
    light_dir = np.array([0.3, 0.3, 0.9])
    light_dir = light_dir / np.linalg.norm(light_dir)
    brightness = np.tensordot(normal, light_dir, axes=([2], [0]))
    brightness = np.clip(brightness, 0.0, 1.0)

    # Base colors from height, then modulate by brightness
    base_colors = cmap_sea(norm_z)
    # Modulate RGB channels (leave alpha intact)
    base_colors[..., :3] = base_colors[..., :3] * (0.5 + 0.6 * brightness[..., None])

    # Add whitecaps on crests
    foam_threshold = 0.55
    foam_mask = Z_new > (foam_threshold * zmax)
    base_colors[foam_mask, :3] = np.clip(base_colors[foam_mask, :3] + 0.45, 0, 1)

    # Créer une NOUVELLE surface avec les couleurs calculées
    current_surf = ax.plot_surface(
        X, Y, Z_new,
        facecolors=base_colors,
        linewidth=0,
        antialiased=True,
        rcount=Ny, ccount=Nx
    )

    # ======================
    # 4️⃣ Animer et orienter le bateau selon la pente locale
    # ======================

    # Calculer nouvelle position du bateau (via la classe Boat)
    x_boat_now, y_boat_now = boat.position_at(t, domain_x=Lx)

    # Bilinéar interpolation pour obtenir la hauteur locale de la mer au centre du bateau
    # Trouver indices entourant la position
    ix = np.searchsorted(x, x_boat_now) - 1
    iy = np.searchsorted(y, y_boat_now) - 1
    ix = np.clip(ix, 0, Nx - 2)
    iy = np.clip(iy, 0, Ny - 2)

    x1, x2 = x[ix], x[ix+1]
    y1, y2 = y[iy], y[iy+1]
    Q11 = Z_new[iy, ix]
    Q21 = Z_new[iy, ix+1]
    Q12 = Z_new[iy+1, ix]
    Q22 = Z_new[iy+1, ix+1]
    if (x2 - x1) == 0 or (y2 - y1) == 0:
        z_center = 0.0
    else:
        tx = (x_boat_now - x1) / (x2 - x1)
        ty = (y_boat_now - y1) / (y2 - y1)
        z_center = (Q11 * (1 - tx) * (1 - ty) + Q21 * tx * (1 - ty) +
                    Q12 * (1 - tx) * ty + Q22 * tx * ty)

    # Calculer pente locale via gradient pour définir les angles de tangage/roulis
    dZ_dy, dZ_dx = np.gradient(Z_new, dy, dx)
    dZdx = dZ_dx[iy, ix]
    dZdy = dZ_dy[iy, ix]

    # Angles petits: pitch autour de l'axe Y (tangage), roll autour de l'axe X (roulis)
    pitch = -np.arctan(dZdx)
    roll = np.arctan(dZdy)

    # Utiliser la méthode de rendu du bateau (classe Boat)
    boat.render(ax, pitch, roll, x_boat_now, y_boat_now, z_center)

    # Retour pour FuncAnimation (on renvoie la surface et l'objet plot du bateau si existant)
    return current_surf, boat._plot

# ======================
# 🔹 FONCTION POUR LANCER L'ANIMATION
# ======================
def start_animation(frames=200, interval=50, show=True):
    """Crée et retourne l'animation. Call `plt.show()` if show=True."""
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=False,
        cache_frame_data=False
    )
    plt.tight_layout()
    if show:
        plt.show()
    return ani


if __name__ == '__main__':
    start_animation()