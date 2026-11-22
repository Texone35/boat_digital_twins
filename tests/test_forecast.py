import numpy as np
import pytest

import jonswap
from boat_presets import PRESETS
from simulation import Simulation
from waveforecast.forecast import SHIP_X, ShipForecast


@pytest.fixture(scope='module')
def run():
    sea = jonswap.SeaState()
    sea.set_params(Hs=2.0, Tp=10.0)
    Z_before = sea.generate_surface(5.0)
    sim = Simulation(sea, PRESETS['Cargo'])
    sim.set_heading(np.radians(60))          # cap oblique : roulis non nul
    fc = ShipForecast(sim)
    for _ in range(200):                     # 10 s : le navire quitte le repos
        sim.step()
        fc.record(sim)
    f = fc.run(sim)
    return sea, sim, fc, f, Z_before


def test_station_keeping_and_frozen_sea(run):
    sea, sim, fc, f, _ = run
    assert sea.frozen and sea.theta_mean == 0.0
    assert abs(sim.state[0] - SHIP_X) < 1.0 and abs(sim.state[6]) < 0.05
    assert not sea.advance(10.0)             # mer figée


def test_forecast_accurate_in_zone_and_degrades_beyond(run):
    _, _, _, f, _ = run
    tau = f.times - f.t0

    def err(j, lo, hi):
        k = (tau >= lo) & (tau < hi)
        return np.sqrt(np.mean((f.predicted[k, j] - f.truth[k, j]) ** 2)) / f.truth[:, j].std()

    for j in (2, 3, 4):                      # heave, roll, pitch
        assert err(j, 0, 20) < 0.15
        assert err(j, f.zone_end, 60) > 2 * err(j, 0, 20)


def test_truth_ghost_matches_twin(run):
    sea, sim, fc, f, _ = run
    t, heave = [], []
    while sim.t < f.times[-1]:
        sim.step()
        t.append(sim.t)
        heave.append(sim.state[2])
    gap = np.interp(t, f.times, f.truth[:, 2]) - np.array(heave)
    assert np.sqrt(np.mean(gap ** 2)) < 0.1 * f.truth[:, 2].std()


def test_unfreeze_restores_sea(run):
    sea, sim, fc, _, Z_before = run
    fc.close(sim)
    assert not sea.frozen
    assert np.array_equal(sea.generate_surface(5.0), Z_before)
