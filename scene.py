"""
Scène 3D PyVista : mer, repères de hauteur, coque, ligne de flottaison et
texte d'infos. Les acteurs sont créés une fois puis mis à jour en place.

Rendu : PyVista (VTK, accéléré GPU). Le tampon de profondeur de la carte
graphique gère l'occlusion mer/bateau pixel par pixel.
"""

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkFlyingEdges2D
from matplotlib import colors as mcolors

import jonswap as sea_mod
from boat import heading_label

# Exagération verticale de l'affichage uniquement (1.0 = échelle réelle).
# ×2 : une houle de 300 m de long pour 20 m de creux paraît presque plate à
# l'échelle réelle. Les graduations (axes, mât, légende) restent en m réels.
Z_SCALE = 2.0
# Opacité de la mer (1.0 = opaque, la carène immergée devient invisible).
SEA_OPACITY = 0.92
# Hs maximal proposé par l'interface : fixe les bornes des axes
HS_MAX = 18.0


def _nice_step(x, steps=(0.25, 0.5, 1.0, 2.0, 5.0, 10.0)):
    """Plus petit pas « rond » ≥ x."""
    for s in steps:
        if s >= x:
            return s
    return steps[-1]


def _rgb_u8(rgba):
    """Couleurs RGB(A) flottantes [0,1] → RGB uint8 attendu par VTK (rgb=True)."""
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255).astype(np.uint8)


def _sea_points(Z, Dx, Dy):
    """Sommets de la mer : grille régulière déplacée horizontalement (choppy)."""
    return np.column_stack([(sea_mod.X + Dx).ravel(), (sea_mod.Y + Dy).ravel(), Z.ravel()])


def _sea_polydata(sea, Z, Dx, Dy):
    """Grille de mer en quads, sommets partagés (ordre C : j*Nx + i)."""
    Nx, Ny = sea_mod.Nx, sea_mod.Ny
    pts = _sea_points(Z, Dx, Dy)
    j, i = np.meshgrid(np.arange(Ny - 1), np.arange(Nx - 1), indexing='ij')
    idx = (j * Nx + i).ravel()
    quads = np.column_stack([np.full_like(idx, 4),
                             idx, idx + 1, idx + Nx + 1, idx + Nx])
    mesh = pv.PolyData(pts, faces=quads.ravel())
    rgb, normals = sea.colors_and_normals(Z, Dx, Dy)
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
        self._img = pv.ImageData(dimensions=(sea_mod.Nx, sea_mod.Ny, 1),
                                 spacing=(sea_mod.dx, sea_mod.dy, 1))
        self._img.point_data['z'] = np.zeros(sea_mod.Nx * sea_mod.Ny)
        self._fe = vtkFlyingEdges2D()
        self._fe.SetInputData(self._img)
        self._fe.SetInputArrayToProcess(0, 0, 0, 0, 'z')
        self.set_levels(levels)
        self.mesh = pv.lines_from_points(np.zeros((2, 3)))
        self.mesh.point_data['niveau0'] = np.zeros(2)

    def set_levels(self, levels):
        self._fe.SetNumberOfContours(len(levels))
        for i, z in enumerate(levels):
            self._fe.SetValue(i, z)

    def update(self, Z, Dx, Dy, lift=0.15):
        self._img.point_data['z'] = Z.ravel()
        self._img.Modified()
        self._fe.Update()
        lines = pv.wrap(self._fe.GetOutput()).copy()
        if lines.n_points:
            x, y = lines.points[:, 0], lines.points[:, 1]
            lines.points = np.column_stack([
                x + sea_mod.interpolate_sea_at_batch(Dx, x, y),
                y + sea_mod.interpolate_sea_at_batch(Dy, x, y),
                sea_mod.interpolate_sea_at_batch(Z, x, y) + lift])
            lines.point_data['niveau0'] = (np.abs(lines.point_data['z']) < 1e-6).astype(float)
        self.mesh.shallow_copy(lines)


def _boat_polydata(tri_verts, tri_colors):
    """Coque en triangles à sommets non partagés : couleur franche par face."""
    n = tri_verts.shape[0]
    faces = np.column_stack([np.full(n, 3), np.arange(3 * n).reshape(n, 3)])
    mesh = pv.PolyData(tri_verts.reshape(-1, 3), faces=faces.ravel())
    mesh.cell_data['rgb'] = _rgb_u8(mcolors.to_rgba_array(tri_colors))
    return mesh


class Scene:
    """Acteurs de la scène et leur mise à jour à chaque frame."""

    def __init__(self, plotter, sim):
        self.plotter = plotter
        self.sim = sim
        sea = sim.sea
        boat = sim.boat
        x0, y0 = sim.state[0], sim.state[1]

        plotter.set_background('#dcebf5', top='#5b87b0')
        plotter.set_scale(zscale=Z_SCALE)

        Z, Dx, Dy = sea.generate_fields(sim.t)
        self.sea_mesh = _sea_polydata(sea, Z, Dx, Dy)
        # Mer légèrement translucide pour laisser deviner la carène immergée.
        # Teinte fixée par SeaState.colors_and_normals ; ombrage et reflets
        # du soleil (spéculaire) calculés par VTK à partir des normales par
        # sommet (pas de filtre smooth_shading : ≈20 ms de rendu par frame).
        plotter.add_mesh(self.sea_mesh, scalars='rgb', rgb=True, opacity=SEA_OPACITY,
                         ambient=0.15, diffuse=0.85,
                         specular=0.6, specular_power=60)

        # ── Repères de hauteur ──
        # Lignes de niveau (iso-élévation) à pas « rond » adapté à Hs,
        # recalculées sur la surface affichée ; niveau 0 (mer calme) en jaune.
        # Légèrement surélevées pour ne pas se confondre avec la surface.
        self.iso = _IsoLines([0.0])
        plotter.add_mesh(self.iso.mesh, scalars='niveau0', cmap=['#dfeef2', '#ffe066'],
                         clim=(0, 1), show_scalar_bar=False, opacity=0.5,
                         line_width=1.5, reset_camera=False)

        # Légende des couleurs de la mer (acteur invisible porteur de la barre,
        # même colormap et même plage que la teinte de la mer)
        self.legend = plotter.add_mesh(
            pv.PolyData(np.zeros((1, 3))), scalars=np.zeros(1),
            cmap=sea_mod._SEA_CMAP, clim=(-1, 1),
            scalar_bar_args=dict(title='Elevation mer (m)', vertical=True,
                                 position_x=0.91, position_y=0.2, height=0.6,
                                 width=0.05, n_labels=5, fmt='%+.1f',
                                 color='white', shadow=True))
        self.legend.SetVisibility(False)

        # Échelle de marée : mât rayé planté au niveau de mer calme (z = 0)
        # à côté de la position initiale du bateau ; graduations adaptées à Hs.
        self._gauge_xy = (x0, y0 - 2.5 * max(boat.width, 20.0))
        self._gauge_actors = []
        self._iso_step = self._label_step = None
        self._legend_hs = None
        self.on_sea_changed()

        tris0, cols0 = boat.compute_mesh(0.0, 0.0, x0, y0, 0.0, 0.0)
        self.boat_mesh = _boat_polydata(tris0, cols0)
        plotter.add_mesh(self.boat_mesh, scalars='rgb', rgb=True, preference='cell',
                         show_edges=True, edge_color='#333333', line_width=0.5)

        # Ligne de flottaison et texte d'infos : créés une seule fois puis mis
        # à jour en place. Un add_mesh/add_text par frame déclencherait à
        # chaque appel un rendu complet de la fenêtre (≈100 ms chacun).
        # Maillage de coque à sommets partagés (≈5× moins de points que
        # boat_mesh) pour le contour de flottaison
        self.waterline_src = pv.PolyData(
            boat.local_vertices,
            faces=np.column_stack([np.full(len(boat.triangles), 3),
                                   boat.triangles]).ravel())
        self.waterline_mesh = pv.lines_from_points(np.zeros((2, 3)))
        self.waterline_actor = plotter.add_mesh(self.waterline_mesh, color='black',
                                                line_width=3, reset_camera=False)
        # Texte en vtkTextActor simple (position relative à la fenêtre) et non
        # position='upper_left' : ce dernier crée un vtkCornerAnnotation qui
        # recherche la taille de police à chaque changement de texte (≈95 ms
        # de rendu par frame).
        self.info_txt = plotter.add_text('', position=(0.01, 0.98), viewport=True,
                                         font_size=10, color='white', shadow=True)
        self.info_txt.prop.SetVerticalJustificationToTop()

        # Bornes des axes fixées sur le Hs maximal de l'interface : pas de
        # reconstruction des axes quand la mer change.
        zb = HS_MAX * 1.2 * Z_SCALE
        plotter.show_grid(xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)', n_zlabels=3,
                          bounds=(0, sea_mod.Lx, 0, sea_mod.Ly, -zb, zb))
        self.recenter_camera()

    # ── Caméra ──
    def recenter_camera(self):
        """
        Caméra basse (18°) centrée sur le bateau, à une distance adaptée à
        sa taille : vues de près du niveau de l'eau, les vagues masquent
        l'horizon et leur hauteur se lit bien mieux que de haut.
        """
        xb, yb = self.sim.boat_xy()
        el, az = np.radians(18), np.radians(-60)
        R = float(np.clip(5.0 * self.sim.params.L, 120.0, 0.75 * sea_mod.Lx))
        self.plotter.camera_position = [
            (xb + R*np.cos(el)*np.cos(az), yb + R*np.cos(el)*np.sin(az), R*np.sin(el)),
            (xb, yb, 0.0), (0, 0, 1)]

    def translate_camera(self, dx, dy):
        """Déplace la caméra avec le bateau (orientation et zoom conservés)."""
        cam = self.plotter.camera
        px, py, pz = cam.position
        fx, fy, fz = cam.focal_point
        cam.position = (px + dx, py + dy, pz)
        cam.focal_point = (fx + dx, fy + dy, fz)

    # ── Mer ──
    def on_sea_changed(self):
        """
        Adapte légende, lignes de niveau et mât de marée au Hs courant.
        Légende : à chaque variation de plus de 2 % ; lignes et mât :
        seulement quand leur pas « rond » change (reconstruction du mât).
        """
        Hs = self.sim.sea.Hs
        if self._legend_hs is None or abs(Hs - self._legend_hs) > 0.02 * self._legend_hs:
            h = sea_mod.color_range(Hs)
            self.legend.mapper.scalar_range = (-h, h)
            self.legend.mapper.lookup_table.scalar_range = (-h, h)
            self._legend_hs = Hs

        iso_step = _nice_step(Hs / 8.0)
        label_step = _nice_step(Hs / 4.0, steps=(0.5, 1.0, 2.0, 5.0, 10.0))
        if iso_step == self._iso_step and label_step == self._label_step:
            return
        self._iso_step, self._label_step = iso_step, label_step

        zmax = iso_step * np.ceil(1.5 * Hs / iso_step)
        self.iso.set_levels(list(np.arange(-zmax, zmax + 1e-9, iso_step)))

        for a in self._gauge_actors:
            self.plotter.remove_actor(a, render=False)
        gx, gy = self._gauge_xy
        zg = label_step * np.ceil(1.2 * Hs / label_step)
        gauge = pv.lines_from_points(
            np.column_stack([np.full(400, gx), np.full(400, gy),
                             np.linspace(-zg, zg, 400)]))
        gauge.point_data['bande'] = (np.floor(gauge.points[:, 2] / iso_step) % 2)
        ticks = np.arange(-zg, zg + 1e-9, label_step)
        fmt = '{:+.0f} m' if label_step >= 1 else '{:+.1f} m'
        self._gauge_actors = [
            self.plotter.add_mesh(gauge, scalars='bande', cmap=['#d62728', '#ffffff'],
                                  show_scalar_bar=False, line_width=9,
                                  render_lines_as_tubes=True, reset_camera=False,
                                  render=False),
            self.plotter.add_point_labels(
                np.column_stack([np.full(ticks.size, gx + 3), np.full(ticks.size, gy),
                                 ticks * Z_SCALE]),
                [fmt.format(z) for z in ticks], font_size=11, text_color='white',
                shape=None, show_points=False, always_visible=True,
                reset_camera=False, render=False),
        ]

    # ── Mise à jour par frame ──
    def update(self, Z, Dx, Dy, wave, frame, n_sub=1, fps=0.0, paused=False):
        sim, sea, boat = self.sim, self.sim.sea, self.sim.boat
        eta, nu = sim.state[:6], sim.state[6:]
        xb, yb = sim.boat_xy()

        # Élévation réelle de la mer sous chaque sommet de la coque, pour
        # une coloration immergé/émergé dynamique (recalculée chaque frame)
        # plutôt qu'un seuil fixe.
        def sea_elev_fn(x_arr, y_arr):
            return sea_mod.interpolate_sea_at_batch(
                Z, x_arr % sea_mod.Lx, y_arr % sea_mod.Ly)

        # Coque en position absolue (heave, roll, pitch issus de Cummins)
        boat_tris, boat_colors = boat.compute_mesh(
            pitch=eta[4], roll=eta[3], x_pos=xb, y_pos=yb,
            z_center=eta[2], yaw=eta[5], sea_elev_fn=sea_elev_fn)

        # Mise à jour en place des maillages (topologie fixe : seuls les
        # sommets et les couleurs changent). Écriture en place dans les
        # tableaux VTK existants (pas de réallocation).
        rgb, normals = sea.colors_and_normals(Z, Dx, Dy)
        self.sea_mesh.points[:] = _sea_points(Z, Dx, Dy)
        self.sea_mesh.point_data['rgb'][:] = _rgb_u8(rgb).reshape(-1, 3)
        self.sea_mesh.point_data['Normals'][:] = normals
        self.sea_mesh.Modified()

        self.boat_mesh.points = boat_tris.reshape(-1, 3)
        self.boat_mesh.cell_data['rgb'] = _rgb_u8(mcolors.to_rgba_array(boat_colors))

        # Lignes de niveau une frame sur deux (elles bougent lentement)
        if frame % 2 == 0 or n_sub > 1:
            self.iso.update(Z, Dx, Dy)

        # Ligne de flottaison réelle : intersection coque / surface de mer
        # (iso-contour 0 de la hauteur des sommets au-dessus de la mer)
        pts = boat.world_vertices
        self.waterline_src.points = pts
        self.waterline_src.point_data['h'] = pts[:, 2] - sea_elev_fn(pts[:, 0], pts[:, 1])
        waterline = self.waterline_src.contour([0.0], scalars='h')
        if waterline.n_points:
            self.waterline_mesh.shallow_copy(waterline)
        self.waterline_actor.SetVisibility(waterline.n_points > 0)

        # Infos temps réel
        mu, V, omega_e = sim.hydro.encounter(sim.state)
        Te = 2*np.pi / omega_e
        p = sim.params
        trans = '   (transition en cours)' if sea.is_transitioning() else ''
        self.info_txt.SetInput(
            f'{p.name}  —  L={p.L:g} m  B={p.B:g} m  T={p.T:g} m  '
            f'{p.mass/1e3:,.0f} t'.replace(',', ' ') + '\n'
            f't = {sim.t:.1f} s    {fps:4.1f} img/s    (echelle verticale x{Z_SCALE:g})'
            f'{"    PAUSE" if paused else ""}\n'
            f'Mer : Hs={sea.Hs:.1f} m  Tp={sea.Tp:.1f} s  gamma={sea.gamma:.1f}  '
            f'dir={round(np.degrees(sea.theta_mean)) % 360}°{trans}\n'
            f'crete max = {Z.max():+.1f} m    creux min = {Z.min():+.1f} m    '
            f'Hs inst. = {4*Z.std():.1f} m\n'
            f'heave = {eta[2]:+.2f} m    houle moy. = {wave[0]:+.2f} m\n'
            f'roll  = {np.degrees(eta[3]):+.2f}°    pitch = {np.degrees(eta[4]):+.2f}°\n'
            f'omega_e = {omega_e:.3f} rad/s    Te = {Te:.1f} s\n'
            f'V = {V:.1f} m/s    cap = {round(np.degrees(eta[5])) % 360}°    '
            f'μ = {np.degrees(mu):.0f}° ({heading_label(mu)})')

    def set_boat(self, boat):
        """Nouvelle coque : même topologie (Nu, Nv fixes), sommets remplacés."""
        self.waterline_src.points = boat.local_vertices
