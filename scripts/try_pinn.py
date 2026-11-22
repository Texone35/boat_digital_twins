"""
Essai du PINN sur UNE fenêtre de mesures, comparé à la baseline LSQ.

    python scripts/try_pinn.py domain=small iters=3000 causal_eps=10
    python scripts/try_pinn.py domain=full iters=20000 adapt_every=100 lbfgs_iters=500

Arguments clé=valeur : champs de PINNConfig (wavepinn/pinn.py), plus
  domain=small|full  sondes à 250-100 m (horizon ≈ 25 s) ou 600-300 m (≈ 60 s)
  seed_sea=7         graine de la mer (Hs 2 m, Tp 10 s, γ 3.3, bruit 3 %)
Chaque essai ajoute une ligne (config + erreurs) à results/pinn_trials.jsonl.
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wavepinn import baseline_lsq, physics, pinn, sensors  # noqa: E402

DOMAINS = {   # sondes (x_far, x_near), fenêtre T, horizon
    'small': dict(probes=(-250.0, -100.0), T=120.0, horizon=45.0),
    'full': dict(probes=(-600.0, -300.0), T=180.0, horizon=90.0),
}


def parse(argv):
    kw = dict(a.split('=', 1) for a in argv)
    dom = DOMAINS[kw.pop('domain', 'small')]
    seed_sea = int(kw.pop('seed_sea', 7))
    cfg = pinn.config_from_kv([f'{k}={v}' for k, v in kw.items()],
                              log_every=500, horizon=dom['horizon'])
    return dom, seed_sea, cfg, kw


def main():
    dom, seed_sea, cfg, raw = parse(sys.argv[1:])
    sea = physics.make_sea_1d(2.0, 10.0, 3.3, seed=seed_sea)
    s = sensors.probes(sea, sensors.probe_positions(4, *dom['probes']), 0, 400,
                       noise=0.03, seed=1)
    t0, T = 300.0, dom['T']
    m = (s['t'] >= t0 - T) & (s['t'] < t0)
    win = (s['t'][m], s['x'], s['eta'][:, m])
    taus = np.arange(-60.0, dom['horizon'], 1.0)
    truth = physics.eta(sea, [0.0], t0 + taus)[:, 0]

    model = pinn.WavePINN(10.0, 3.3, 4.0 * float(np.std(win[2])), cfg)
    print(f'device {model.dev} — {cfg}', flush=True)
    start = time.time()
    model.fit(*win)
    train_s = time.time() - start
    p = model.predict([0.0], t0 + taus)[:, 0]
    b = baseline_lsq.forecaster(10.0, 3.3)(*win)([0.0], t0 + taus)[:, 0]
    zone_end = physics.predictable_window(0.0, T, -s['x'], *physics.energy_band(10.0, 3.3))[1]

    def rms(q, k):
        return round(float(np.sqrt(np.mean((q[k] - truth[k]) ** 2)) / sea['Hs']), 4)

    bins = {'past': taus < 0, 'zone': (taus >= 0) & (taus <= zone_end),
            'beyond': taus > zone_end}
    res = dict(domain=raw | {'probes': dom['probes']}, train_s=round(train_s),
               device=str(model.dev), zone_end_s=round(zone_end, 1),
               pinn={k: rms(p, v) for k, v in bins.items()},
               lsq={k: rms(b, v) for k, v in bins.items()},
               final_losses=dict(zip(('it', 'data', 'kin', 'dyn', 'lap', 'fond', 'w_phys',
                                      'causal_min'), model.history[-1])),
               config=dataclasses.asdict(cfg))
    fit = model.predict(s['x'], win[0]).T - win[2]
    res['probe_fit'] = round(float(np.sqrt(np.mean(fit ** 2)) / sea['Hs']), 4)
    print(json.dumps({k: res[k] for k in ('train_s', 'probe_fit', 'zone_end_s', 'pinn',
                                          'lsq')}, indent=1))
    out = ROOT / 'results' / 'pinn_trials.jsonl'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
