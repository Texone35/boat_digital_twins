import numpy as np
import pytest

torch = pytest.importorskip('torch')
from wavepinn import pinn  # noqa: E402


def test_airy_wave_has_zero_residuals():
    """Onde d'Airy en unités réduites : ω̃² = 2π k̃ (eau profonde)
         η = cos(kx − ωt),  φ = (2π/ω) sin(kx − ωt) e^{kz}
    → résidus cinématique, dynamique et Laplace nuls ; fond ≈ e^{−kh}."""
    w = 1.3 * 2 * np.pi
    k = w**2 / (2 * np.pi)
    h = 1.0

    def eta_net(x, t):
        return torch.cos(k * x - w * t)

    def phi_net(x, t, z):
        return (2 * np.pi / w) * torch.sin(k * x - w * t) * torch.exp(k * z)

    def pts(*shape):
        return torch.rand(*shape, dtype=torch.float64).requires_grad_()

    n = 256
    zi = (-h * torch.rand(n, 1, dtype=torch.float64)).requires_grad_()
    r = pinn.residuals(eta_net, phi_net, pts(n, 1), pts(n, 1), pts(n, 1), pts(n, 1), zi,
                       pts(n, 1), pts(n, 1), h)
    for res in r[:3]:
        assert res.abs().max().item() < 1e-9
    assert r[3].abs().max().item() < 2 * k * np.exp(-k * h) * (2 * np.pi / w)


def test_config_from_kv():
    cfg = pinn.config_from_kv(['iters=12', 'causal_eps=2.5', 'linear_skip=true'], horizon=30.0)
    assert (cfg.iters, cfg.causal_eps, cfg.linear_skip, cfg.horizon) == (12, 2.5, True, 30.0)
    with pytest.raises(KeyError):
        pinn.config_from_kv(['nope=1'])


def test_fit_runs_with_all_options():
    from wavepinn import physics, sensors
    sea = physics.make_sea_1d(2.0, 10.0, 3.3, seed=3)
    s = sensors.probes(sea, sensors.probe_positions(2, -200.0, -100.0), 0.0, 60.0)
    cfg = pinn.PINNConfig(iters=8, n_colloc=64, hidden=16, depth=2, n_ff=16, phys_ramp=2,
                          adapt_every=2, causal_eps=5.0, causal_chunks=4, lbfgs_iters=3,
                          ff_mode='dispersion', ff_sigma=0.1, device='cpu', horizon=20.0)
    model = pinn.WavePINN(10.0, 3.3, 2.0, cfg).fit(s['t'], s['x'], s['eta'])
    out = model.predict([0.0, 50.0], [70.0, 71.0, 72.0])
    assert out.shape == (3, 2) and np.isfinite(out).all()
    assert np.isfinite(model.history[-1][1:]).all()


def test_fourier_modes_travel_towards_plus_x():
    for mode in ('box', 'dispersion'):
        cfg = pinn.PINNConfig(ff_mode=mode, ff_sigma=0.1, n_ff=64, device='cpu')
        B = pinn.WavePINN(10.0, 3.3, 2.0, cfg).B
        assert (B[1] < 0).all()                       # arg = 2π(bₓx − |bₜ|t)
    with pytest.raises(ValueError):
        pinn.WavePINN(10.0, 3.3, 2.0, pinn.PINNConfig(ff_mode='nope', device='cpu'))
