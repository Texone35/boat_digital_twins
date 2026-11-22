"""
Génère les mers 1D et les mesures des sondes, plus data/splits/index.csv.

    python scripts/make_dataset.py [--n-probes 4] [--noise 0.03]

Grille : Hs ∈ {1, 2, 4} m × Tp ∈ {7, 10, 13} s × γ ∈ {1, 3.3}, 3 graines par
case. La graine d'indice 0 / 1 / 2 va en train / val / test : chaque état
de mer est présent dans chaque split, et deux splits ne partagent aucune mer.
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waveforecast import data, physics, sensors  # noqa: E402

HS = (1.0, 2.0, 4.0)
TP = (7.0, 10.0, 13.0)
GAMMA = (1.0, 3.3)
SPLITS = ('train', 'val', 'test')
DURATION = 600.0   # s de mesures par mer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-probes', type=int, default=4)
    ap.add_argument('--noise', type=float, default=0.03, help='écart-type du bruit / Hs')
    ap.add_argument('--fs', type=float, default=2.0, help='fréquence des sondes (Hz)')
    args = ap.parse_args()

    rows = []
    for case, (Hs, Tp, gamma) in enumerate(itertools.product(HS, TP, GAMMA)):
        for j, split in enumerate(SPLITS):
            seed = 10 * case + j
            sea = physics.make_sea_1d(Hs, Tp, gamma, seed)
            data.save_sea(sea)
            s = sensors.probes(sea, sensors.probe_positions(args.n_probes), 0.0, DURATION,
                               fs=args.fs, noise=args.noise, seed=seed)
            path = data.save_sensors(sea, s)
            rows.append(dict(id=data.sea_id(sea), Hs=Hs, Tp=Tp, gamma=gamma, seed=seed,
                             sensors=path.name, split=split))

    index = data.DATA / 'splits' / 'index.csv'
    index.parent.mkdir(parents=True, exist_ok=True)
    with open(index, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} mers -> {data.DATA}')


if __name__ == '__main__':
    main()
