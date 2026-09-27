"""
Main runner — animation 3D mer JONSWAP + navire Cummins 6-DDL.

Couplage mer → bateau à chaque frame :
  - élévation η(x_b,y_b,t)     → amplitude locale effective
  - pentes ∂η/∂x, ∂η/∂y       → moments Froude-Krylov roll/pitch

Position du bateau : heave, roll, pitch (eta[2:5]) sont ABSOLUS, par
rapport au niveau de mer calme. La houle agit via les forces de
Froude-Krylov calculées sur la flottaison (boat.waterplane_wave).

Rendu : PyVista (VTK, accéléré GPU). Le tampon de profondeur de la carte
graphique gère l'occlusion mer/bateau pixel par pixel : plus besoin de
fusionner mer et coque dans une seule collection comme avec mplot3d.
"""

import time

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkFlyingEdges2D
from matplotlib import colors as mcolors

from boat import (Boat, step_state, waterplane_wave, omega_p, wave_heading,
                  encounter_frequency, heading_angle, L, B, T)
import jonswap as sea

# Exagération verticale de l'affichage uniquement (1.0 = échelle réelle).
# ×2 : une houle de 300 m de long pour 20 m de creux paraît presque plate à
# l'échelle réelle. Les graduations (axes, mât, légende) restent en m réels.
Z_SCALE = 2.0
# Opacité de la mer (1.0 = opaque, la carène immergée devient invisible).
SEA_OPACITY = 0.92
# Écart entre deux lignes de niveau dessinées sur la mer (m).
ISO_STEP = 2.0
# Pas de physique max. par frame affichée pour rattraper le temps réel
# quand une frame prend plus de dt à calculer et afficher.
MAX_SUBSTEPS = 4


def _rgb_u8(rgba):
    """Couleurs RGB(A) flottantes [0,1] → RGB uint8 attendu par VTK (rgb=True)."""
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255).astype(np.uint8)


def _sea_points(Z, Dx, Dy):
    """Sommets de la mer : grille régulière déplacée horizontalement (choppy)."""
    return np.column_stack([(sea.X + Dx).ravel(), (sea.Y + Dy).ravel(), Z.ravel()])


def _sea_polydata(Z, Dx, Dy):
    """Grille de mer en quads, sommets partagés (ordre C : j*Nx + i)."""
    pts = _sea_points(Z, Dx, Dy)
    j, i = np.meshgrid(np.arange(sea.Ny - 1), np.arange(sea.Nx - 1), indexing='ij')
    idx = (j * sea.Nx + i).ravel()
    quads = np.column_stack([np.full_like(idx, 4),
                             idx, idx + 1, idx + sea.Nx + 1, idx + sea.Nx])
    mesh = pv.PolyData(pts, faces=quads.ravel())
    rgb, normals = sea.get_sea_colors_and_normals(Z, Dx, Dy)
    mesh.point_data['rgb'] = _rgb_u8(rgb).reshape(-1, 3)
    mesh.point_data['Normals'] = normals
    mesh.point_data.active_normals_name = 'Normals'
    return mesh


class _IsoLines:
    """
    Lignes de niveau de la mer (iso-élévation), calculées sur la grille
    régulière avec vtkFlyingEdges2D (≈10× plus rapide qu'un contour sur le
    maillage déplacé), puis déplacées comme la surface affichée (choppy).
    """

    def __init__(self, levels):
        self._img = pv.ImageData(dimensions=(sea.Nx, sea.Ny, 1),
                                 spacing=(sea.dx, sea.dy, 1))
        self._img.point_data['z'] = np.zeros(sea.Nx * sea.Ny)
        self._fe = vtkFlyingEdges2D()
        self._fe.SetInputData(self._img)
        self._fe.SetNumberOfContours(len(levels))
        for i, z in enumerate(levels):
            self._fe.SetValue(i, z)
        self._fe.SetInputArrayToProcess(0, 0, 0, 0, 'z')
        self.mesh = pv.lines_from_points(np.zeros((2, 3)))
        self.mesh.point_data['niveau0'] = np.zeros(2)

    def update(self, Z, Dx, Dy, lift=0.15):
        self._img.point_data['z'] = Z.ravel()
        self._img.Modified()
        self._fe.Update()
        lines = pv.wrap(self._fe.GetOutput()).copy()
        if lines.n_points:
            x, y = lines.points[:, 0], lines.points[:, 1]
            lines.points = np.column_stack([
                x + sea.interpolate_sea_at_batch(Dx, x, y),
                y + sea.interpolate_sea_at_batch(Dy, x, y),
                sea.interpolate_sea_at_batch(Z, x, y) + lift])
            lines.point_data['niveau0'] = (np.abs(lines.point_data['z']) < 1e-6).astype(float)
        self.mesh.shallow_copy(lines)


def _boat_polydata(tri_verts, tri_colors):
    """Coque en triangles à sommets non partagés : couleur franche par face."""
    n = tri_verts.shape[0]
    faces = np.column_stack([np.full(n, 3), np.arange(3 * n).reshape(n, 3)])
    mesh = pv.PolyData(tri_verts.reshape(-1, 3), faces=faces.ravel())
    mesh.cell_data['rgb'] = _rgb_u8(mcolors.to_rgba_array(tri_colors))
    return mesh


def main():
    print(f'Sea + Boat (Cummins 6-DOF)  Hs={sea.Hs}m  Tp={sea.Tp}s')

    # ── Bateau ── maillage à l'échelle VISUAL_SCALE de la géométrie physique
    # (L, B, T). Garder 1.0 : la houle n'est pas mise à l'échelle, donc une
    # coque réduite (franc-bord compris) serait submergée par des vagues
    # dimensionnées pour le vrai navire.
    VISUAL_SCALE = 1.0
    x0, y0 = sea.Lx/2, sea.Ly/2
    boat = Boat(length=L*VISUAL_SCALE, width=B*VISUAL_SCALE, draft=T*VISUAL_SCALE,
                x0=x0, y0=y0, speed=2.5,
                Nu=41, Nv=15, color='#FFD700', underwater_color='#2E8B57')

    # ── État initial ──────────────────────────────
    # η = [surge(x), sway(y), heave, roll, pitch, yaw]
    # ν = [u,        v,       w,     p,    q,     r  ]
    dt    = 0.05
    state = np.zeros(12, dtype=float)
    state[0] = x0      # position x initiale
    state[1] = y0      # position y initiale
    state[6] = 2.5     # vitesse surge (m/s)

    # ── Scène ─────────────────────────────────────
    plotter = pv.Plotter(window_size=(1400, 800),
                         title=f'JONSWAP + Cummins 6-DDL  —  Hs={sea.Hs}m  '
                               f'Tp={sea.Tp}s  γ={sea.gamma}')
    plotter.set_background('#dcebf5', top='#5b87b0')
    plotter.set_scale(zscale=Z_SCALE)

    sea_mesh = _sea_polydata(*sea.generate_sea_fields(0.0))
    # Mer légèrement translucide pour laisser deviner la carène immergée.
    # Teinte fixée par get_sea_colors ; ombrage et reflets du soleil
    # (spéculaire) calculés par VTK à partir des normales par sommet,
    # fournies par sea.get_sea_colors_and_normals (pas de filtre
    # smooth_shading : ≈20 ms de rendu par frame).
    plotter.add_mesh(sea_mesh, scalars='rgb', rgb=True, opacity=SEA_OPACITY,
                     ambient=0.15, diffuse=0.85,
                     specular=0.6, specular_power=60)

    # ── Repères de hauteur ──
    # Lignes de niveau (iso-élévation) tous les ISO_STEP m, recalculées à
    # chaque frame sur la surface affichée ; niveau 0 (mer calme) plus épais.
    # Légèrement surélevées pour ne pas se confondre avec la surface.
    zmax_iso = ISO_STEP * np.ceil(1.5 * sea.Hs / ISO_STEP)
    iso_levels = list(np.arange(-zmax_iso, zmax_iso + 1e-9, ISO_STEP))
    iso = _IsoLines(iso_levels)
    plotter.add_mesh(iso.mesh, scalars='niveau0', cmap=['#dfeef2', '#ffe066'],
                     clim=(0, 1), show_scalar_bar=False, opacity=0.5,
                     line_width=1.5, reset_camera=False)

    # Légende des couleurs de la mer (acteur invisible porteur de la barre,
    # même colormap et même plage que sea.get_sea_colors)
    h_range = 0.6 * sea.Hs * sea.wind_scale
    legend = plotter.add_mesh(
        pv.PolyData(np.zeros((1, 3))), scalars=np.zeros(1),
        cmap=sea._SEA_CMAP, clim=(-h_range, h_range),
        scalar_bar_args=dict(title='Elevation mer (m)', vertical=True,
                             position_x=0.91, position_y=0.2, height=0.6,
                             width=0.05, n_labels=5, fmt='%+.0f',
                             color='white', shadow=True))
    legend.SetVisibility(False)

    # Échelle de marée : mât rayé tous les 2 m, gradué tous les 5 m, planté
    # au niveau de mer calme (z = 0) à côté de la position initiale du bateau.
    gx, gy = x0, y0 - 2.5 * B
    zg = 5.0 * np.ceil(1.2 * sea.Hs / 5.0)
    gauge = pv.lines_from_points(
        np.column_stack([np.full(400, gx), np.full(400, gy),
                         np.linspace(-zg, zg, 400)]))
    gauge.point_data['bande'] = (np.floor(gauge.points[:, 2] / 2.0) % 2)
    plotter.add_mesh(gauge, scalars='bande', cmap=['#d62728', '#ffffff'],
                     show_scalar_bar=False, line_width=9,
                     render_lines_as_tubes=True)
    ticks = np.arange(-zg, zg + 1e-9, 5.0)
    plotter.add_point_labels(
        np.column_stack([np.full(ticks.size, gx + 3), np.full(ticks.size, gy),
                         ticks * Z_SCALE]),
        [f'{z:+.0f} m' for z in ticks], font_size=11, text_color='white',
        shape=None, show_points=False, always_visible=True)

    tris0, cols0, _ = boat.compute_mesh(0.0, 0.0, x0, y0, 0.0, 0.0)
    boat_mesh = _boat_polydata(tris0, cols0)
    plotter.add_mesh(boat_mesh, scalars='rgb', rgb=True, preference='cell',
                     show_edges=True, edge_color='#333333', line_width=0.5)

    # Ligne de flottaison et texte d'infos : créés une seule fois puis mis
    # à jour en place. Un add_mesh/add_text par frame déclencherait à
    # chaque appel un rendu complet de la fenêtre (≈100 ms chacun).
    # Maillage de coque à sommets partagés (≈5× moins de points que
    # boat_mesh) pour le contour de flottaison
    waterline_src = pv.PolyData(
        boat.local_vertices,
        faces=np.column_stack([np.full(len(boat.triangles), 3),
                               boat.triangles]).ravel())
    waterline_mesh = pv.lines_from_points(np.zeros((2, 3)))
    waterline_actor = plotter.add_mesh(waterline_mesh, color='black',
                                       line_width=3, reset_camera=False)
    # Texte en vtkTextActor simple (position relative à la fenêtre) et non
    # position='upper_left' : ce dernier crée un vtkCornerAnnotation qui
    # recherche la taille de police à chaque changement de texte (≈95 ms
    # de rendu par frame).
    info_txt = plotter.add_text('', position=(0.01, 0.98), viewport=True,
                                font_size=10, color='white', shadow=True)
    info_txt.prop.SetVerticalJustificationToTop()

    plotter.show_grid(xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)', n_zlabels=3,
                      bounds=(0, sea.Lx, 0, sea.Ly,
                              -sea.Hs*1.5*Z_SCALE, sea.Hs*1.5*Z_SCALE))

    # Caméra basse (12°) et rapprochée, centrée sur le bateau : vues de
    # près du niveau de l'eau, les vagues masquent l'horizon et leur
    # hauteur se lit bien mieux que de haut.
    el, az, R = np.radians(18), np.radians(-60), 0.75 * sea.Lx
    focus = (x0, y0, 0.0)
    plotter.camera_position = [
        (focus[0] + R*np.cos(el)*np.cos(az),
         focus[1] + R*np.cos(el)*np.sin(az),
         focus[2] + R*np.sin(el)),
        focus, (0, 0, 1)]

    # ── Boucle d'animation ────────────────────────
    def update(frame, n_sub=1):
        t = frame * dt

        # 1. Surface de mer au temps t (Z : élévation pour la physique ;
        # Dx, Dy : déplacements horizontaux pour l'affichage)
        Z, Dx, Dy = sea.generate_sea_fields(t)

        # 2. Position du bateau (repliement périodique dans le domaine)
        xb = state[0] % sea.Lx
        yb = state[1] % sea.Ly

        # 3. Houle sous la flottaison : élévation moyenne + pentes
        wave = waterplane_wave(Z, xb, yb, state[5])

        # 4. Intégration Cummins RK4 (n_sub pas de dt pour rattraper le temps
        # réel ; la houle de la frame est réutilisée pour ces sous-pas)
        for k in range(n_sub):
            state[:] = step_state(state, t - (n_sub - k) * dt, dt, wave=wave)

        eta = state[:6]
        nu  = state[6:]

        # Élévation réelle de la mer sous chaque sommet de la coque, pour
        # une coloration immergé/émergé dynamique (recalculée chaque frame)
        # plutôt qu'un seuil fixe.
        def sea_elev_fn(x_arr, y_arr):
            return sea.interpolate_sea_at_batch(
                Z, x_arr % sea.Lx, y_arr % sea.Ly)

        # 5. Coque en position absolue (heave, roll, pitch issus de Cummins)
        boat_tris, boat_colors, _ = boat.compute_mesh(
            pitch=eta[4], roll=eta[3], x_pos=xb, y_pos=yb,
            z_center=eta[2], yaw=eta[5], sea_elev_fn=sea_elev_fn)

        # 6. Mise à jour en place des maillages (topologie fixe : seuls les
        # sommets et les couleurs changent)
        # Écriture en place dans les tableaux VTK existants (pas de
        # réallocation) ; pyvista marque chaque tableau modifié.
        rgb, normals = sea.get_sea_colors_and_normals(Z, Dx, Dy)
        sea_mesh.points[:] = _sea_points(Z, Dx, Dy)
        sea_mesh.point_data['rgb'][:] = _rgb_u8(rgb).reshape(-1, 3)
        sea_mesh.point_data['Normals'][:] = normals
        sea_mesh.Modified()

        boat_mesh.points = boat_tris.reshape(-1, 3)
        boat_mesh.cell_data['rgb'] = _rgb_u8(mcolors.to_rgba_array(boat_colors))

        # Lignes de niveau sur la surface affichée
        # Lignes de niveau une frame sur deux (elles bougent lentement)
        if frame % 2 == 0 or n_sub > 1:
            iso.update(Z, Dx, Dy)

        # Ligne de flottaison réelle : intersection coque / surface de mer
        # (iso-contour 0 de la hauteur des sommets au-dessus de la mer)
        pts = boat.world_vertices
        waterline_src.points = pts
        waterline_src.point_data['h'] = pts[:, 2] - sea_elev_fn(pts[:, 0], pts[:, 1])
        waterline = waterline_src.contour([0.0], scalars='h')
        if waterline.n_points:
            waterline_mesh.shallow_copy(waterline)
        waterline_actor.SetVisibility(waterline.n_points > 0)

        # 7. Infos temps réel
        mu      = heading_angle(eta[5], wave_heading)
        V       = abs(nu[0])
        omega_e = encounter_frequency(omega_p, V, mu)
        Te      = 2*np.pi / omega_e
        info_txt.SetInput(
            f't = {t:.1f} s    (echelle verticale x{Z_SCALE:g})\n'
            f'crete max = {Z.max():+.1f} m    creux min = {Z.min():+.1f} m    '
            f'(ecart {Z.max() - Z.min():.1f} m)    Hs inst. = {4*Z.std():.1f} m\n'
            f'heave = {eta[2]:+.2f} m    houle moy. = {wave[0]:+.2f} m\n'
            f'roll  = {np.degrees(eta[3]):+.2f}°    pitch = {np.degrees(eta[4]):+.2f}°\n'
            f'omega_e = {omega_e:.3f} rad/s    Te = {Te:.1f} s\n'
            f'V = {V:.1f} m/s    μ = {np.degrees(mu):.0f}°')

    # ── Boucle d'animation directe (pas de timer VTK) ──
    # Les timers Windows ont une granularité ≈16 ms et passent après les
    # autres messages : avec eux on plafonnait à ~13 images/s. Ici chaque
    # tour traite les événements (souris, fermeture), calcule et affiche.
    # Le temps simulé suit l'horloge réelle : si une frame a pris plus de
    # dt, la physique avance de plusieurs pas (MAX_SUBSTEPS au plus).
    closing = {'v': False}
    plotter.iren.add_observer('ExitEvent', lambda *_: closing.update(v=True))

    update(0)
    plotter.show(interactive_update=True, auto_close=False)
    t0, frame = time.perf_counter(), 0
    while True:
        plotter.iren.process_events()
        if closing['v']:
            break
        target = int((time.perf_counter() - t0) / dt)
        if target <= frame:
            time.sleep(0.002)          # en avance sur le temps réel
            continue
        if target - frame > MAX_SUBSTEPS:
            # Trop en retard : on décale l'origine plutôt que d'accumuler
            t0 += (target - frame - MAX_SUBSTEPS) * dt
            target = frame + MAX_SUBSTEPS
        n_sub, frame = target - frame, target
        update(frame, n_sub)
        plotter.render()
    plotter.close()

if __name__ == '__main__':
    main()
