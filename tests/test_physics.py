import numpy as np
import pytest
from scipy.signal import hilbert

from wavepinn import data, physics, sensors


@pytest.fixture(scope='module')
def sea():
    return physics.make_sea_1d(Hs=2.0, Tp=10.0, gamma=3.3, seed=1)


def test_hs_exact_and_measured(sea):
    assert 4 * np.sqrt(np.sum(sea['A']**2) / 2) == pytest.approx(2.0)
    Z = physics.eta(sea, np.arange(0, 4000, 4.0), np.arange(0, 1200, 4.0))
    # Fenêtre finie : battements lents entre fréquences voisines → ~3 % d'écart attendu
    assert 4 * Z.std() == pytest.approx(2.0, rel=0.05)


def test_same_seed_same_sea():
    a, b = (physics.make_sea_1d(1.0, 7.0, 1.0, seed=5) for _ in range(2))
    c = physics.make_sea_1d(1.0, 7.0, 1.0, seed=6)
    assert np.array_equal(a['phi'], b['phi']) and np.array_equal(a['omega'], b['omega'])
    assert not np.array_equal(a['phi'], c['phi'])


def test_group_velocity_is_half_phase_velocity(sea):
    c = sea['omega'] / sea['k']
    np.testing.assert_allclose(physics.group_velocity(sea['omega']), c / 2)


def test_packet_travels_at_group_velocity():
    # Paquet étroit centré en ω0, crête de l'enveloppe en (x=0, t=0)
    w0, sw = 0.8, 0.03
    w = np.linspace(w0 - 4 * sw, w0 + 4 * sw, 400)
    pkt = dict(A=np.exp(-0.5 * ((w - w0) / sw) ** 2), omega=w, k=w**2 / physics.g,
               phi=np.zeros_like(w))
    d = 600.0
    t = np.arange(0, 250, 0.1)
    env = np.abs(hilbert(physics.eta(pkt, [d], t)[:, 0]))
    assert t[env.argmax()] == pytest.approx(d / physics.group_velocity(w0), rel=0.02)


def test_energy_band_holds_90_percent(sea):
    lo, hi = physics.energy_band(sea['Tp'], sea['gamma'])
    wp = 2 * np.pi / sea['Tp']
    assert (lo / wp, hi / wp) == pytest.approx((0.83, 1.99), abs=0.02)
    inside = (sea['omega'] >= lo) & (sea['omega'] <= hi)
    assert np.sum(sea['A'][inside]**2) / np.sum(sea['A']**2) == pytest.approx(0.90, abs=0.03)


def test_predictable_window_matches_plan_table():
    # Tp = 10 s : 60 s d'horizon demandent d ≈ 562 m et une fenêtre ≥ 83 s
    lo, hi = physics.energy_band(10.0, 3.3)
    start, end = physics.predictable_window(0.0, 90.0, 562.0, lo, hi)
    assert end == pytest.approx(60.0, abs=1.0) and start < end
    assert physics.predictable_window(0.0, 80.0, 562.0, lo, hi) is None
    # Plusieurs sondes : l'union va jusqu'à la plus lointaine
    _, end2 = physics.predictable_window(0.0, 200.0, [300.0, 562.0], lo, hi)
    assert end2 == pytest.approx(end)


def test_probes_shape_and_noise(sea):
    x = sensors.probe_positions(4)
    s = sensors.probes(sea, x, 0.0, 300.0, fs=2.0, noise=0.05, seed=3)
    assert s['eta'].shape == (4, 600) and s['eta'].dtype == np.float32
    resid = s['eta'] - physics.eta(sea, x, s['t']).T
    assert resid.std() == pytest.approx(0.05 * sea['Hs'], rel=0.1)
    assert sensors.probe_positions(1).tolist() == [sensors.X_NEAR]


def test_npz_round_trip(sea, tmp_path):
    loaded = data.load_sea(data.save_sea(sea, root=tmp_path))
    assert loaded['seed'] == sea['seed'] and loaded['Hs'] == sea['Hs']
    np.testing.assert_array_equal(loaded['phi'], sea['phi'])
    s = sensors.probes(sea, sensors.probe_positions(3), 0.0, 10.0)
    p = data.save_sensors(sea, s, root=tmp_path)
    assert p.name == 'sea_Hs2.0_Tp10.0_g3.3_seed001_probes3_noise0.03.npz'
    assert data.load_sensors(p)['sea_id'] == data.sea_id(sea)
