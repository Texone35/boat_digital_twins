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
from matplotlib.animation import FuncAnimation

from boat import (Boat, step_state, omega_p, wave_heading,
                  Hs, Tp, encounter_frequency, heading_angle)
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

    # ── Bateau ────────────────────────────────────
    x0, y0 = sea.Lx/2, sea.Ly/2
    boat = Boat(length=20.0, width=5.0, height=3.0,
                x0=x0, y0=y0, speed=2.5,
                Nu=25, Nv=9, color='#FFD700')

    # ── État initial ──────────────────────────────
    # η = [surge(x), sway(y), heave, roll, pitch, yaw]
    # ν = [u,        v,       w,     p,    q,     r  ]
    dt    = 0.05
    state = np.zeros(12, dtype=float)
    state[0] = x0      # position x initiale
    state[1] = y0      # position y initiale
    state[6] = 2.5     # vitesse surge (m/s)

    current_surf = [None]
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

        # 5. Rendu de la mer
        if current_surf[0] is not None:
            current_surf[0].remove()
        current_surf[0] = ax.plot_surface(
            sea.X, sea.Y, Z,
            facecolors=sea.get_sea_colors(Z),
            linewidth=0, antialiased=True,
            rcount=sea.Ny, ccount=sea.Nx)

        # 6. Rendu du bateau
        # eta[2] = heave RELATIF à la surface libre (Cummins)
        # z_center = position absolue = mer + déplacement dynamique
        z_center = elev + eta[2]
        boat.render(ax, pitch=eta[4], roll=eta[3],
                    x_pos=xb, y_pos=yb,
                    z_center=z_center, yaw=eta[5])

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

        return current_surf[0],

    ani = FuncAnimation(fig, update, frames=400,
                        interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()