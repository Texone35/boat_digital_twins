"""
Stockage .npz. Règles :
  - data/raw/     : composantes de la mer (la vérité se régénère par physics.eta)
  - data/sensors/ : mesures seules → ce que voit la prévision
  - la graine est dans le nom ; les splits train/val/test se font par mer.
"""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'


def sea_id(sea):
    return (f"sea_Hs{sea['Hs']:.1f}_Tp{sea['Tp']:.1f}_g{sea['gamma']:.1f}"
            f"_seed{sea['seed']:03d}")


def _save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def _load(path):
    with np.load(path) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


def save_sea(sea, root=DATA):
    return _save(Path(root) / 'raw' / f'{sea_id(sea)}.npz', sea)


def save_sensors(sea, sensors, root=DATA):
    name = f"{sea_id(sea)}_probes{len(sensors['x'])}_noise{sensors['noise']:.2f}.npz"
    return _save(Path(root) / 'sensors' / name, dict(sensors, sea_id=sea_id(sea)))


load_sea = load_sensors = _load
