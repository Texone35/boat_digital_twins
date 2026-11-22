"""
Identification du roulis d'un navire « réel » aux paramètres cachés, par trois
méthodes, en fonction du bruit de la centrale inertielle.

    python scripts/run_shipid.py            # erreur d'équation + erreur de sortie (≈ 10 s)
    python scripts/run_shipid.py --pinn     # + PINN (≈ 2 min par niveau de bruit, CPU)

Navire réel : Cargo avec GM_T = 0.8 m (nominal 1.2), ζ = 0.05 (nominal 0.10),
b_q = 1 (nominal 0), à poste fixe, houle de travers Hs 4 m, Tp 15 s, 300 s de
mesures à 10 Hz. Sorties : results/shipid/metrics.json, results/shipid/identification.png.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boat_presets import PRESETS  # noqa: E402
from shipid import identify, records  # noqa: E402

NOMINAL = PRESETS['Cargo']
REAL = replace(NOMINAL, GM_T=0.8, zeta_roll=0.05, roll_quad=1.0)
NOISES = (0.0, 0.2, 0.5, 1.0)          # bruit du roulis mesuré (°)
SEA = dict(Hs=4.0, Tp=15.0, heading_deg=90.0, duration=300.0)
OUT = ROOT / 'results' / 'shipid'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pinn', action='store_true', help='inclure le PINN (lent)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    metrics, gm = {}, {}
    for noise in NOISES:
        r = records.record_roll(REAL, noise_deg=noise, **SEA)
        nominal = dict(records.roll_truth(NOMINAL, r['sea']), b_q=0.0)
        found = {'erreur d\'équation': identify.equation_error(r['t'], r['roll'], r['phi_w'])}
        found['erreur de sortie'] = identify.output_error(r['t'], r['roll'], r['phi_w'],
                                                         found['erreur d\'équation'])
        if args.pinn:
            found['PINN'] = identify.pinn(r['t'], r['roll'], r['phi_w'], nominal)
        truth = r['truth']
        metrics[f'bruit {noise}°'] = row = {
            'vrai': dict(GM_T=truth['GM_T'], zeta=round(truth['zeta'], 4), b_q=truth['b_q']),
            'nominal': dict(GM_T=NOMINAL.GM_T, zeta=round(nominal['zeta'], 4), b_q=0.0)}
        for name, m in found.items():
            g = records.gm_from_omega(m['omega_n'], NOMINAL, r['sea'])
            row[name] = dict(GM_T=round(g, 3), zeta=round(m['zeta'], 4), b_q=round(m['b_q'], 2),
                             secondes=m['seconds'])
            gm.setdefault(name, []).append(g)
        if noise == NOISES[-1]:
            example = (r, found['erreur de sortie'])
        print(f'bruit {noise}° : ' + ', '.join(f"{k} GM={v['GM_T']}" for k, v in row.items()
                                              if 'GM_T' in v), flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[2, 1]))
    r, m = example
    k = r['t'] <= 120
    sim = identify.simulate(r['t'], r['phi_w'], m['omega_n'], m['zeta'], m['b_q'],
                            m['phi0'], m['dphi0'])
    a1.plot(r['t'][k], np.degrees(r['roll'][k]), color='0.75', lw=0.8, label='mesuré (bruit 1°)')
    a1.plot(r['t'][k], np.degrees(r['roll_clean'][k]), 'k', lw=1.5, label='roulis réel')
    a1.plot(r['t'][k], np.degrees(sim[k]), 'C0--', lw=1.5, label='modèle identifié')
    a1.set(xlabel='t (s)', ylabel='roulis (°)', title='Roulis : mesure, réalité et modèle identifié')
    a1.legend(fontsize=8)
    for i, (name, g) in enumerate(gm.items()):
        a2.plot(NOISES, g, 'o-', color=f'C{i + 1}', label=name)
    a2.axhline(REAL.GM_T, color='k', lw=1, label='GM_T réel')
    a2.axhline(NOMINAL.GM_T, color='k', ls=':', lw=1, label='GM_T nominal')
    a2.set(xlabel='bruit du roulis mesuré (°)', ylabel='GM_T identifié (m)', title='GM_T vs bruit')
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'identification.png', dpi=130)
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                      encoding='utf-8')


if __name__ == '__main__':
    main()
