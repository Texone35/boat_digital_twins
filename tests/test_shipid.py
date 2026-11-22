from dataclasses import replace

import numpy as np
import pytest

from boat_presets import PRESETS
from shipid import identify, records

NOMINAL = PRESETS['Cargo']
REAL = replace(NOMINAL, GM_T=0.8, zeta_roll=0.05, roll_quad=1.0)


@pytest.fixture(scope='module')
def rec():
    return records.record_roll(REAL, Hs=4.0, Tp=15.0, duration=300.0, noise_deg=0.2)


def test_one_dof_roll_model_matches_twin(rec):
    tr = rec['truth']
    v0 = (rec['roll_clean'][1] - rec['roll_clean'][0]) / (rec['t'][1] - rec['t'][0])
    sim = identify.simulate(rec['t'], rec['phi_w'], tr['omega_n'], tr['zeta'], tr['b_q'],
                            rec['roll_clean'][0], v0)
    assert np.sqrt(np.mean((sim - rec['roll_clean']) ** 2)) < 0.03 * rec['roll_clean'].std()


def test_default_boat_unchanged_by_roll_damping_fields():
    assert (NOMINAL.zeta_roll, NOMINAL.roll_quad) == (0.10, 0.0)
    assert replace(NOMINAL, zeta_roll=0.0).validate()      # ζ = 0 refusé


def test_output_error_recovers_hidden_parameters(rec):
    ee = identify.equation_error(rec['t'], rec['roll'], rec['phi_w'])
    oe = identify.output_error(rec['t'], rec['roll'], rec['phi_w'], ee)
    gm = records.gm_from_omega(oe['omega_n'], NOMINAL, rec['sea'])
    assert gm == pytest.approx(REAL.GM_T, rel=0.02)
    assert oe['zeta'] == pytest.approx(rec['truth']['zeta'], rel=0.15)
    assert oe['b_q'] == pytest.approx(REAL.roll_quad, rel=0.3)


def test_pinn_runs():
    pytest.importorskip('torch')
    r = records.record_roll(REAL, duration=60.0, noise_deg=0.2)
    nominal = dict(records.roll_truth(NOMINAL, r['sea']), b_q=0.0)
    m = identify.pinn(r['t'], r['roll'], r['phi_w'], nominal, iters=5, n_ff=8, hidden=8)
    assert np.isfinite([m['omega_n'], m['zeta'], m['b_q']]).all() and m['fit'].shape == r['t'].shape


def test_calibration_reproduces_hidden_ship():
    from shipid import calibrate
    real = calibrate.hidden_ship(NOMINAL, seed=3)
    r = records.record_roll(real, Hs=4.0, Tp=15.0, noise_deg=0.2)
    cal = calibrate.calibrated_params(NOMINAL, r['sea'],
                                      calibrate.identify_roll(r['t'], r['roll'], r['phi_w']))
    a, b = records.roll_truth(real, r['sea']), records.roll_truth(cal, r['sea'])
    assert cal.GM_T == pytest.approx(real.GM_T, rel=0.02)
    assert b['omega_n'] == pytest.approx(a['omega_n'], rel=0.01)
    assert b['zeta'] == pytest.approx(a['zeta'], rel=0.1)


def test_roll_monitor_window():
    from shipid import calibrate
    m = calibrate.RollMonitor()
    for k in range(int(100 / 0.05)):
        m.record(k * 0.05, np.sin(k * 0.05), 0.0)
    assert not m.ready(100.0)                          # < MIN_WINDOW de données
    for k in range(int(100 / 0.05), int(400 / 0.05)):
        m.record(k * 0.05, np.sin(k * 0.05), 0.0)
    assert m.duration() <= calibrate.WINDOW and m.ready(400.0)
    t, roll, _ = m.start(400.0)
    assert np.allclose(np.diff(t), 1 / calibrate.FS)
    assert not m.ready(405.0) and m.ready(400.0 + calibrate.PERIOD)


def test_operator_units_and_training_step():
    pytest.importorskip('torch')
    from shipid import operator
    roll, phi_w, params = operator.synthetic_batch(16, np.random.default_rng(0))
    assert roll.shape == (16, operator.N_T)
    np.testing.assert_allclose(operator._from_unit(operator._to_unit(params)), params)
    model, info = operator.train(n_train=64, n_val=16, epochs=1, batch=32, log=None)
    p = operator.predict(model, roll[0], phi_w[0])
    assert np.isfinite([p['omega_n'][0], p['zeta'][0], p['b_q'][0]]).all()
