"""
Main runner — animation 3D mer JONSWAP + navire Cummins 6-DDL.

Couplage mer → bateau à chaque frame :
  - élévation η(x_b,y_b,t)     → amplitude locale effective
  - pentes ∂η/∂x, ∂η/∂y       → moments Froude-Krylov roll/pitch

Position verticale du bateau :
  z_center = sea_elev + eta[2]
  sea_elev = surface libre locale (mer)
  eta[2]   = heave = déplacement RELATIF à la surface libre
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

from boat import (Boat, step_state, omega_p, wave_heading,
                  Hs, Tp, encounter_frequency, heading_angle, L, B, T)
import jonswap as sea


def main():
    print(f'Sea + Boat (Cummins 6-DOF)  Hs={sea.Hs}m  Tp={sea.Tp}s')

    # ── Figure ────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)', fontsize=11, labelpad=8)
    ax.set_ylabel('Y (m)', fontsize=11, labelpad=8)
    ax.set_zlabel('Z (m)', fontsize=11, labelpad=8)
    ax.set_xlim(0, sea.Lx)
    ax.set_ylim(0, sea.Ly)
    ax.set_zlim(-sea.Hs * 2.5, sea.Hs * 2.5)
    ax.set_title(
        f'JONSWAP + Cummins 6-DDL  —  Hs={sea.Hs}m  Tp={sea.Tp}s  γ={sea.gamma}',
        fontsize=13, pad=18)
    ax.view_init(elev=25, azim=-60)
    # mplot3d ne trie par profondeur qu'À L'INTÉRIEUR d'une même collection
    # de polygones, jamais entre deux artistes différents (il ne compare
    # alors que leur profondeur moyenne globale). Pour une occlusion locale
    # correcte entre la mer et le bateau — une vague plus proche de la
    # caméra doit cacher la partie du bateau derrière elle, et inversement —
    # les deux sont fusionnés en une seule Poly3DCollection à chaque frame
    # (voir plus bas). computed_zorder reste à sa valeur par défaut (True).

    # ── Bateau ── maillage à l'échelle visuelle VISUAL_SCALE de la géométrie
    # physique (L, B, T restent inchangés pour la dynamique 6 DDL : seul le
    # rendu est réduit, pour rester lisible dans le domaine de mer 150×150 m)
    VISUAL_SCALE = 0.5
    x0, y0 = sea.Lx/2, sea.Ly/2
    boat = Boat(length=L*VISUAL_SCALE, width=B*VISUAL_SCALE, draft=T*VISUAL_SCALE,
                x0=x0, y0=y0, speed=2.5,
                Nu=25, Nv=9, color='#FFD700', underwater_color='#2E8B57')

    # ── État initial ──────────────────────────────
    # η = [surge(x), sway(y), heave, roll, pitch, yaw]
    # ν = [u,        v,       w,     p,    q,     r  ]
    dt    = 0.05
    state = np.zeros(12, dtype=float)
    state[0] = x0      # position x initiale
    state[1] = y0      # position y initiale
    state[6] = 2.5     # vitesse surge (m/s)

    scene_mesh = [None]
    waterline_artist = [None]
    info_txt = [ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                           fontsize=9, color='white',
                           bbox=dict(facecolor='black', alpha=0.55, pad=3))]

    # ── Boucle d'animation ────────────────────────
    def update(frame):
        t = frame * dt

        # 1. Surface de mer au temps t
        Z = sea.generate_sea_surface(t)

        # 2. Position du bateau (repliement périodique dans le domaine)
        xb = state[0] % sea.Lx
        yb = state[1] % sea.Ly

        # 3. Données de mer locales sous la coque
        elev       = sea.interpolate_sea_at(Z, xb, yb)
        dzdx, dzdy = sea.sea_gradient_at(Z, xb, yb)

        # 4. Intégration Cummins RK4
        new_state = step_state(state, t, dt,
                               sea_elev=elev,
                               sea_dzdx=dzdx,
                               sea_dzdy=dzdy)
        state[:] = new_state

        eta = state[:6]
        nu  = state[6:]

        # 5. Position verticale du bateau
        # eta[2] = heave RELATIF à la surface libre (Cummins)
        # z_center = position absolue = mer + déplacement dynamique
        #
        # Lissage du niveau de mer affiché sur la longueur de coque (proue/
        # milieu/poupe) : évite que le bateau "claque" verticalement en
        # suivant l'élévation d'un seul point, alors que la coque a une
        # emprise réelle de plusieurs dizaines de mètres.
        psi = eta[5]
        half_L = L / 2.0
        bow_x   = (xb + half_L*np.cos(psi)) % sea.Lx
        bow_y   = (yb + half_L*np.sin(psi)) % sea.Ly
        stern_x = (xb - half_L*np.cos(psi)) % sea.Lx
        stern_y = (yb - half_L*np.sin(psi)) % sea.Ly
        elev_bow   = sea.interpolate_sea_at(Z, bow_x, bow_y)
        elev_stern = sea.interpolate_sea_at(Z, stern_x, stern_y)
        elev_render = 0.25*elev_bow + 0.5*elev + 0.25*elev_stern

        z_center = elev_render + eta[2]

        # Élévation réelle de la mer sous chaque sommet de la coque, pour
        # une coloration immergé/émergé dynamique (recalculée chaque frame)
        # plutôt qu'un seuil fixe.
        def sea_elev_fn(x_arr, y_arr):
            return sea.interpolate_sea_at_batch(
                Z, x_arr % sea.Lx, y_arr % sea.Ly)

        boat_tris, boat_colors, waterline = boat.compute_mesh(
            pitch=eta[4], roll=eta[3], x_pos=xb, y_pos=yb,
            z_center=z_center, yaw=eta[5], sea_elev_fn=sea_elev_fn)

        # 6. Fusion mer + bateau dans une seule Poly3DCollection pour une
        # occlusion locale correcte (priorité au visuel de la mer : toute
        # vague plus proche de la caméra cache la partie du bateau qu'elle
        # recouvre, avec sa couleur — voir commentaire sur ax.view_init).
        sea_tris, sea_colors = sea.build_sea_mesh(Z)
        all_tris = np.concatenate([sea_tris, boat_tris], axis=0)
        facecolors = list(map(tuple, sea_colors)) + list(boat_colors)
        edgecolors = ['none'] * len(sea_tris) + ['black'] * len(boat_tris)

        if scene_mesh[0] is not None:
            scene_mesh[0].remove()
        scene_mesh[0] = Poly3DCollection(
            all_tris, facecolor=facecolors, edgecolor=edgecolors,
            linewidth=1.2, antialiased=True)
        ax.add_collection3d(scene_mesh[0])

        if waterline_artist[0] is not None:
            waterline_artist[0].remove()
        if waterline is not None:
            xs, ys, zs = waterline
            waterline_artist[0] = ax.plot(xs, ys, zs, color='k', linewidth=2)[0]
        else:
            waterline_artist[0] = None

        # 7. Infos temps réel
        mu      = heading_angle(eta[5], wave_heading)
        V       = abs(nu[0])
        omega_e = encounter_frequency(omega_p, V, mu)
        Te      = 2*np.pi / omega_e
        info_txt[0].set_text(
            f't = {t:.1f} s\n'
            f'heave = {eta[2]:+.2f} m    elev = {elev:+.2f} m\n'
            f'roll  = {np.degrees(eta[3]):+.2f}°    pitch = {np.degrees(eta[4]):+.2f}°\n'
            f'ω_e = {omega_e:.3f} rad/s    Te = {Te:.1f} s\n'
            f'V = {V:.1f} m/s    μ = {np.degrees(mu):.0f}°')

        return scene_mesh[0],

    ani = FuncAnimation(fig, update, frames=400,
                        interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()