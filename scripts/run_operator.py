"""
Opérateur neuronal (FNO) contre PINN et erreur de sortie pour identifier le roulis.

    python scripts/run_shipid.py --pinn   # d'abord : résultats PINN de référence (≈ 7 min)
    python scripts/run_operator.py        # entraîne l'opérateur une fois (≈ 4 min CPU), compare
    python scripts/run_operator.py        # ensuite : recharge results/shipid/roll_fno.pt
    python scripts/run_operator.py --pinn 10   # + PINN sur les 10 premiers navires de B (≈ 17 min)

A. Les 4 enregistrements de §6.3 (Cargo caché, bruit 0 à 1°) : erreur de sortie,
   PINN (repris de results/shipid/metrics.json) et opérateur.
B. 30 navires cachés tirés au hasard (Cargo, porte-conteneurs, frégate), mers
   aléatoires, bruit 0.5° : erreur de sortie et opérateur ; PINN sur les N premiers
   (--pinn N), et statistiques des trois méthodes sur ce même sous-ensemble.
C. Opérateur sur 1000 exemples synthétiques jamais vus (généralisation).
Sorties : results/shipid/operator.json, results/shipid/operator.png.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from boat_presets import PRESETS  # noqa: E402
from shipid import calibrate, identify, operator, records  # noqa: E402

OUT = ROOT / 'results' / 'shipid'
WEIGHTS = OUT / 'roll_fno.pt'
REF_SEA = dict(Hs=4.0, Tp=15.0, heading_deg=90.0)
NOISES = (0.0, 0.2, 0.5, 1.0)


def fno_identify(model, rec):
    """Opérateur sur un enregistrement du jumeau (10 Hz → 1 Hz)."""
    start = time.time()
    p = operator.predict(model, operator.decimate(rec['roll'], 0.1),
                         operator.decimate(rec['phi_w'], 0.1))
    return dict(omega_n=p['omega_n'][0], zeta=p['zeta'][0], b_q=p['b_q'][0],
                seconds=round(time.time() - start, 4))


def summary(params, sea, m):
    return dict(GM_T=round(float(records.gm_from_omega(m['omega_n'], params, sea)), 3),
                zeta=round(float(m['zeta']), 4), b_q=round(float(m['b_q']), 2),
                secondes=m['seconds'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--pinn', type=int, default=0, help='PINN sur les N premiers navires de B')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if WEIGHTS.exists() and not args.retrain:
        model, info = operator.load(WEIGHTS)
    else:
        model, info = operator.train()
        operator.save(model, WEIGHTS, info)
    print('entraînement :', info, flush=True)
    res = dict(entrainement=info)

    # A. Enregistrements de référence
    cargo = PRESETS['Cargo']
    real = replace(cargo, GM_T=0.8, zeta_roll=0.05, roll_quad=1.0)
    ref = OUT / 'metrics.json'
    pinn_ref = json.loads(ref.read_text(encoding='utf-8')) if ref.exists() else {}
    res['A'] = {}
    for noise in NOISES:
        r = records.record_roll(real, noise_deg=noise, **REF_SEA)
        key = f'bruit {noise}°'
        res['A'][key] = row = dict(vrai=dict(GM_T=0.8, zeta=round(r['truth']['zeta'], 4), b_q=1.0))
        row['erreur de sortie'] = summary(cargo, r['sea'], calibrate.identify_roll(
            r['t'], r['roll'], r['phi_w']))
        if 'PINN' in pinn_ref.get(key, {}):
            row['PINN'] = pinn_ref[key]['PINN']
        row['opérateur'] = summary(cargo, r['sea'], fno_identify(model, r))
        print(key, {k: v for k, v in row.items()}, flush=True)

    # B. Navires cachés aléatoires
    rng = np.random.default_rng(123)
    err = {'erreur de sortie': [], 'opérateur': [], 'PINN': []}
    timing = {k: [] for k in err}
    for i in range(30):
        nominal = PRESETS[('Cargo', 'Porte-conteneurs', 'Frégate')[i % 3]]
        real = calibrate.hidden_ship(nominal, seed=1000 + i)
        r = records.record_roll(real, Hs=rng.uniform(2, 5), Tp=rng.uniform(8, 18),
                                heading_deg=rng.uniform(60, 120), noise_deg=0.5, seed=i)
        tr = r['truth']
        found = [('erreur de sortie', calibrate.identify_roll(r['t'], r['roll'], r['phi_w'])),
                 ('opérateur', fno_identify(model, r))]
        if i < args.pinn:
            start = dict(records.roll_truth(nominal, r['sea']), b_q=0.0)
            found.append(('PINN', identify.pinn(r['t'], r['roll'], r['phi_w'], start)))
            print(f'navire {i} : PINN fait', flush=True)
        for name, m in found:
            gm = records.gm_from_omega(m['omega_n'], nominal, r['sea'])
            err[name].append([gm / real.GM_T - 1, m['zeta'] / tr['zeta'] - 1, m['b_q'] - tr['b_q']])
            timing[name].append(m['seconds'])
    err = {k: np.array(v) for k, v in err.items() if v}

    def stats(e, t):
        a = np.abs(e)
        return dict(n=len(e), GM_T_erreur_mediane_pct=round(100 * float(np.median(a[:, 0])), 2),
                    GM_T_erreur_max_pct=round(100 * float(np.max(a[:, 0])), 2),
                    zeta_erreur_mediane_pct=round(100 * float(np.median(a[:, 1])), 1),
                    zeta_erreur_max_pct=round(100 * float(np.max(a[:, 1])), 1),
                    b_q_erreur_mediane=round(float(np.median(a[:, 2])), 2),
                    b_q_erreur_max=round(float(np.max(a[:, 2])), 2),
                    secondes_par_enregistrement=round(float(np.mean(t)), 4))
    res['B'] = {name: stats(e, timing[name]) for name, e in err.items()}
    if args.pinn:
        n = args.pinn
        res['B_sous_ensemble'] = {name: stats(e[:n], timing[name][:n]) for name, e in err.items()}
    print('B :', res['B'], flush=True)

    # C. Généralisation sur données synthétiques jamais vues
    roll, phi_w, params = operator.synthetic_batch(1000, np.random.default_rng(999))
    p = operator.predict(model, roll, phi_w)
    rel_w = np.abs(p['omega_n'] / params[:, 0] - 1)
    res['C'] = dict(omega_n_erreur_mediane_pct=round(100 * float(np.median(rel_w)), 2),
                    zeta_erreur_mediane_pct=round(
                        100 * float(np.median(np.abs(p['zeta'] / params[:, 1] - 1))), 1),
                    b_q_erreur_mediane=round(float(np.median(np.abs(p['b_q'] - params[:, 2]))), 2))
    print('C :', res['C'], flush=True)
    (OUT / 'operator.json').write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                       encoding='utf-8')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, j, label in zip(axes, range(3), ('GM_T : erreur relative', 'ζ : erreur relative',
                                             'b_q : erreur absolue')):
        data = [err[n][:, j] * (100 if j < 2 else 1) for n in err]
        ax.boxplot(data, tick_labels=[f'{n}\n({len(err[n])} navires)' for n in err])
        ax.tick_params(labelsize=8)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(label + (' (%)' if j < 2 else ''), fontsize=10)
    fig.suptitle('Navires cachés tirés au hasard, bruit 0.5° : erreur d\'identification par méthode')
    fig.tight_layout()
    fig.savefig(OUT / 'operator.png', dpi=130)


if __name__ == '__main__':
    main()
