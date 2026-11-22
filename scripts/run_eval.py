"""
Évalue une méthode de prédiction sur les mers d'un split (data/splits/index.csv).

    python scripts/make_dataset.py
    python scripts/run_eval.py --method lsq
    python scripts/run_eval.py --method pinn --t0-step 60 iters=20000 causal_eps=5   # GPU

Sorties dans results/<méthode>/ :
  metrics.json   horizon utile (seuil 20 % Hs) vs horizon théorique, RMS dans / hors zone
  curves.npz     RMS/Hs et corrélation vs horizon, par Tp
  error_map.png  carte (x, t) de l'erreur + zone théorique (mer Hs 2 m, Tp 10 s, γ 3.3)
et results/horizon.png : toutes les méthodes déjà évaluées, superposées.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wavepinn import baseline_lsq, data, evaluate, physics  # noqa: E402

T_WIN = 180.0                               # fenêtre de mesure (s)
TAUS = np.arange(-60.0, 120.0, 0.5)         # horizons (τ < 0 : reconstruction du passé)
ZONE_FRAC = 0.90                            # bande d'énergie de la zone théorique
STYLE = {'lsq': '-', 'pinn': '--'}


def load(split):
    with open(data.DATA / 'splits' / 'index.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split'] == split:
                yield (data.load_sea(data.DATA / 'raw' / f"{row['id']}.npz"),
                       data.load_sensors(data.DATA / 'sensors' / row['sensors']))


def make_forecaster(args, sea):
    if args.method == 'lsq':
        return baseline_lsq.forecaster(sea['Tp'], sea['gamma'])
    from wavepinn import pinn          # torch importé seulement si besoin
    return pinn.forecaster(sea['Tp'], sea['gamma'], pinn.config_from_kv(args.cfg))


def theoretical_horizon(sens, Tp, gamma):
    lo, hi = physics.energy_band(Tp, gamma, ZONE_FRAC)
    return physics.predictable_window(0.0, T_WIN, -sens['x'], lo, hi)[1]


def plot_horizons(results):
    """Superpose les courbes de toutes les méthodes présentes dans results/*/curves.npz."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for path in sorted(results.glob('*/curves.npz')):
        method, c = path.parent.name, np.load(path)
        for i, Tp in enumerate(c['Tp']):
            col = f'C{i}'
            a1.plot(c['taus'], c['rms'][i], col, ls=STYLE.get(method, ':'),
                    label=f'{method}, Tp = {Tp:g} s')
            a2.plot(c['taus'], c['corr'][i], col, ls=STYLE.get(method, ':'))
            for a in (a1, a2):
                a.axvline(c['zone'][i], color=col, ls=':', lw=1)
    a1.axhline(evaluate.USEFUL_THRESHOLD, color='k', ls=':', lw=1)
    a1.set(ylabel='RMS / Hs', title='Erreur au navire vs horizon (pointillés : fin de zone '
           'théorique ; seuil 20 % Hs)')
    a2.set(xlabel='horizon τ = t − t₀ (s)', ylabel='corrélation', ylim=(0, 1.02))
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / 'horizon.png', dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=('lsq', 'pinn'), default='lsq')
    ap.add_argument('--split', default='test')
    ap.add_argument('--t0-step', type=float, default=20.0, help='pas entre deux prévisions (s)')
    ap.add_argument('cfg', nargs='*', help='PINN : surcharges clé=valeur de PINNConfig')
    args = ap.parse_args()
    out = ROOT / 'results' / args.method
    out.mkdir(parents=True, exist_ok=True)
    t0s = np.arange(T_WIN, 600.0, args.t0_step)

    by_tp, start = {}, time.time()
    for sea, sens in load(args.split):
        pred, truth = evaluate.forecast_errors(sea, sens, make_forecaster(args, sea),
                                               T_WIN, t0s, TAUS)
        g = by_tp.setdefault(sea['Tp'], dict(pred=[], truth=[], zone=[]))
        g['pred'].append(pred)
        g['truth'].append(truth)
        g['zone'].append(theoretical_horizon(sens, sea['Tp'], sea['gamma']))
        print(f"{data.sea_id(sea)}  ({time.time() - start:.0f} s)", flush=True)

    curves = dict(taus=TAUS, Tp=[], rms=[], corr=[], zone=[])
    metrics = {}
    for Tp, g in sorted(by_tp.items()):
        rms, corr = evaluate.horizon_metrics(np.vstack(g['pred']), np.vstack(g['truth']))
        zone = float(np.mean(g['zone']))
        inside = (TAUS >= 0) & (TAUS <= zone)
        metrics[f'Tp={Tp:g}'] = dict(
            n_seas=len(g['pred']), n_forecasts=sum(len(p) for p in g['pred']),
            useful_horizon_s=evaluate.useful_horizon(TAUS, rms),
            theoretical_horizon_s=round(zone, 1),
            rms_in_zone=round(float(rms[inside].mean()), 4),
            rms_beyond_zone=round(float(rms[TAUS > zone + 10].mean()), 4))
        for k, v in (('Tp', Tp), ('rms', rms), ('corr', corr), ('zone', zone)):
            curves[k].append(v)
    np.savez(out / 'curves.npz', **curves)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    plot_horizons(out.parent)

    # Carte (x, t) pour une mer de test représentative
    sea, sens = next((s, m) for s, m in load(args.split)
                     if (s['Hs'], s['Tp'], s['gamma']) == (2.0, 10.0, 3.3))
    t0 = 400.0
    m = (sens['t'] >= t0 - T_WIN) & (sens['t'] < t0)
    f = make_forecaster(args, sea)(sens['t'][m], sens['x'], sens['eta'][:, m])
    x, t = np.arange(-800.0, 400.0, 5.0), np.arange(t0 - T_WIN, t0 + 120.0, 0.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = evaluate.plot_error_map(ax, evaluate.error_map(sea, f, x, t), x, t, t0, T_WIN,
                                 sens['x'], *physics.energy_band(sea['Tp'], sea['gamma'],
                                                                 ZONE_FRAC))
    fig.colorbar(im, ax=ax, label='RMS locale / Hs')
    ax.set_title(f"{args.method.upper()} — {data.sea_id(sea)}")
    fig.tight_layout()
    fig.savefig(out / 'error_map.png', dpi=130)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
