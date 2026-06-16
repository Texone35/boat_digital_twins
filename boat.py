import numpy as np
from matplotlib import colors as mcolors

def create_boat_mesh(length=20.0, width=5.0, height=3.0, Nu=25, Nv=9, color='#FFD700'):
    """
    Génère une coque paramétrique centrée à l'origine.

    Returns:
        boat_local: (N,3) array of vertices centered on origin
        triangles: (M,3) array of triangle indices
        color: hex color string for the hull
    """
    u = np.linspace(-length/2, length/2, Nu)
    v = np.linspace(-width/2, width/2, Nv)
    U, V = np.meshgrid(u, v, indexing='ij')

    deck_height = height
    hull_depth = 1.8

    long_factor = 1 - (2 * U / length) ** 2
    long_factor = np.clip(long_factor, 0.0, 1.0)

    trans_factor = 1 - (2 * np.abs(V) / width) ** 2
    trans_factor = np.clip(trans_factor, 0.0, 1.0)

    Z = deck_height - hull_depth * (long_factor * trans_factor)

    # Place the waterline at z = 0 (so parts of the hull are below 0)
    waterline = deck_height - hull_depth * 0.5
    Z = Z - waterline

    boat_vertices = np.vstack([
        U.flatten(),
        V.flatten(),
        Z.flatten()
    ]).T

    triangles = []
    for i in range(Nu - 1):
        for j in range(Nv - 1):
            idx = i * Nv + j
            triangles.append([idx, idx + 1, idx + Nv])
            triangles.append([idx + 1, idx + Nv + 1, idx + Nv])
    boat_triangles = np.array(triangles)

    # Validate color
    try:
        _ = mcolors.to_rgb(color)
    except Exception:
        color = '#8B4513'

    return boat_vertices, boat_triangles, color


class Boat:
    """Classe représentant le bateau: géométrie locale, couleur, position et rendu."""
    def __init__(self, length=20.0, width=5.0, height=3.0, x0=0.0, y0=0.0,
                 speed=1.0, Nu=25, Nv=9, color='#8B4513'):
        self.length = length
        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0
        self.speed = speed
        self.local_vertices, self.triangles, self.color = create_boat_mesh(
            length=length, width=width, height=height, Nu=Nu, Nv=Nv, color=color
        )
        # Plots to remove between frames
        self._plot = None
        self._waterline = None

    def position_at(self, t, domain_x=None):
        x = (self.x0 + self.speed * t)
        if domain_x is not None and domain_x > 0:
            x = x % domain_x
        return x, self.y0

    def render(self, ax, pitch, roll, x_pos, y_pos, z_center):
        """Applique rotation/translation et affiche le bateau + ligne de flottaison.

        ax: Axes3D
        pitch: angle (rad) rotation autour Y
        roll: angle (rad) rotation autour X
        x_pos, y_pos, z_center: position globale
        """
        # Supprimer anciens plots
        if self._plot is not None:
            try:
                self._plot.remove()
            except Exception:
                pass
        if self._waterline is not None:
            try:
                self._waterline.remove()
            except Exception:
                pass

        # Matrices de rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R = Ry @ Rx

        transformed = (R @ self.local_vertices.T).T + np.array([x_pos, y_pos, z_center])

        # Tracer la coque avec la couleur fournie par create_boat_mesh
        self._plot = ax.plot_trisurf(
            transformed[:, 0], transformed[:, 1], transformed[:, 2],
            triangles=self.triangles,
            color=self.color,
            alpha=1.0,
            linewidth=1.2,
            edgecolor='black',
            antialiased=True
        )

        # Tracer la ligne de flottaison approximative
        tol = max(0.05, 0.02 * self.height)
        mask = np.abs(transformed[:, 2] - z_center) < tol
        pts = transformed[mask]
        if pts.shape[0] >= 3:
            centroid = pts[:, :2].mean(axis=0)
            angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
            order = np.argsort(angles)
            pts_ord = pts[order]
            xs = np.append(pts_ord[:,0], pts_ord[0,0])
            ys = np.append(pts_ord[:,1], pts_ord[0,1])
            zs = np.append(pts_ord[:,2], pts_ord[0,2])
            # store the Line3D object
            self._waterline = ax.plot(xs, ys, zs, color='k', linewidth=2)[0]
        else:
            self._waterline = None
