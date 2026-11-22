import numpy as np

from wavepinn import baseline_lsq, evaluate, physics, sensors


def test_lsq_accurate_inside_zone_and_fails_beyond():
    sea = physics.make_sea_1d(Hs=2.0, Tp=10.0, gamma=3.3, seed=11)
    sens = sensors.probes(sea, sensors.probe_positions(4), 0.0, 400.0, noise=0.0)
    T, taus = 180.0, np.arange(0.0, 110.0, 1.0)
    pred, truth = evaluate.forecast_errors(sea, sens, baseline_lsq.forecaster(10.0, 3.3),
                                           T, [200.0, 260.0, 320.0], taus)
    rms, _ = evaluate.horizon_metrics(pred, truth)
    end = physics.predictable_window(0.0, T, -sens['x'], *physics.energy_band(10.0, 3.3))[1]
    assert rms[taus <= end - 20].mean() < 0.05     # loin de la borne : quasi exact
    assert rms[taus >= end + 20].mean() > 0.15     # hors zone : information jamais mesurée


def test_zone_bounds_empty_upstream_of_probes():
    lo, hi = evaluate.zone_bounds(np.array([-700.0, 0.0]), [-600.0, -300.0], 0.0, 180.0,
                                  *physics.energy_band(10.0, 3.3))
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    assert lo[1] < 0.0 < hi[1]
