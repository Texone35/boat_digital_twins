"""
Mesures synthétiques d'un navire « réel » : le jumeau tourne avec des
caractéristiques cachées (BoatParams), à poste fixe sur une mer JONSWAP, et
on enregistre ce que verraient ses capteurs :
  - roulis φ(t) (centrale inertielle, bruitée) ;
  - pente transversale de houle sous la flottaison, φ_w = atan(s_y)
    (mesure de houle : radar, ou reconstruction waveforecast).

Équation du roulis vue par l'identification (1 DDL, poste fixe) :
    φ̈ + 2ζω_n φ̇ + ω_n²(φ − φ_w) + b_q |φ̇| φ̇ = 0
"""

import numpy as np

import jonswap
from boat import HydroModel, rho, g


def roll_truth(params, sea):
    """(ω_n, ζ, b_q) exacts du navire `params` à poste fixe (ω_e = ω_p), et son inertie."""
    hydro = HydroModel(params, sea)
    we = sea.omega_p                              # vitesse nulle : ω_e = ω_p
    inertia = hydro.M_diag[3] + hydro.A_diag(we)[3]
    C44 = hydro.C_mat[3, 3]
    return dict(omega_n=float(np.sqrt(C44 / inertia)),
                zeta=float(hydro.B_diag(we)[3] / (2 * np.sqrt(C44 * inertia))),
                b_q=float(params.roll_quad), GM_T=float(params.GM_T), inertia=float(inertia))


def gm_from_omega(omega_n, params, sea):
    """GM_T = ω_n²·(Ixx + A44) / (ρ g ∇), l'inertie étant supposée connue (plans du navire)."""
    return omega_n**2 * roll_truth(params, sea)['inertia'] / (rho * g * params.nabla)


def record_roll(params, Hs=3.0, Tp=13.0, heading_deg=90.0, duration=300.0, fs=10.0,
                noise_deg=0.2, input_noise=0.0, seed=0, spin_up=60.0, dt=0.05):
    """
    Simule le navire réel `params` (poste fixe, cap heading_deg, houle vers +x)
    et retourne dict(t, roll, phi_w, roll_clean, truth) échantillonnés à fs.
    noise_deg : bruit du roulis mesuré ; input_noise : bruit relatif sur φ_w.
    """
    sea = jonswap.SeaState(seed=seed)
    sea.set_params(Hs=Hs, Tp=Tp, theta_mean=0.0)
    hydro = HydroModel(params, sea)
    hydro.U_target = 0.0
    state = np.zeros(12)
    state[0], state[1], state[5] = jonswap.Lx / 2, jonswap.Ly / 2, np.radians(heading_deg)

    n = int(round((spin_up + duration) / dt))
    times = dt * np.arange(1, n + 1)
    # shortcut: points de flottaison fixes (poste fixe, pas de dérive : surge/sway non excités)
    xs, ys = hydro.waterplane_points(state[0], state[1], state[5])
    waves = np.column_stack(hydro.wave_plane(sea.eta_points(xs, ys, times)))

    every = int(round(1.0 / (fs * dt)))
    t_out, roll, phi_w = [], [], []
    for i in range(n):
        # houle échantillonnée en fin de pas, tenue sur le pas (comme Simulation.step)
        state = hydro.step_state(state, times[i] - dt, dt, wave=tuple(waves[i]))
        if times[i] > spin_up and (i + 1) % every == 0:
            t_out.append(times[i] - spin_up)
            roll.append(state[3])
            phi_w.append(np.arctan(waves[i, 2]))

    rng = np.random.default_rng(seed + 1)
    roll, phi_w = np.array(roll), np.array(phi_w)
    return dict(t=np.array(t_out),
                roll=roll + rng.normal(0, np.radians(noise_deg), roll.shape),
                phi_w=phi_w + rng.normal(0, input_noise * phi_w.std(), phi_w.shape),
                roll_clean=roll, truth=roll_truth(params, sea), sea=sea)
