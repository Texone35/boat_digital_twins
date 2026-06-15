import numpy as np
from matplotlib import colors as mcolors

def create_boat_mesh(length=20.0, width=5.0, height=3.0, Nu=25, Nv=9, color='#8B4513'):
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
