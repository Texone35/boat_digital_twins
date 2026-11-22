"""
PINN en potentiel (option 3) : houle linéaire 1D, eau profonde.

Deux réseaux : η_θ(x, t) et φ_θ(x, z, t). Perte = écart aux sondes + résidus
de la théorie potentielle linéarisée, calculés par dérivation automatique.

Adimensionnement (x, z par λp ; t par Tp ; η par σ_η = Hs/4 ; φ par
σ_η·λp/Tp ; origine des temps en t0 = fin de la fenêtre de mesure) :
    Laplace       φ_xx + φ_zz = 0            −h ≤ z ≤ 0
    cinématique   η_t − φ_z   = 0            z = 0
    dynamique     φ_t + 2π η  = 0            z = 0   (g·Tp²/λp = 2π)
    fond          φ_z         = 0            z = −h
d'où la dispersion ω̃² = 2π k̃ tanh(k̃ h̃). Fond à h = λ_max/2 (λ de la plus
longue composante reconstruite) : tanh(kh) ≥ 0.996, donc eau profonde.

Le réseau tourne sur GPU si disponible (Colab), sinon CPU.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from wavepinn.physics import energy_band, g

TWO_PI = 2 * np.pi


@dataclass
class PINNConfig:
    hidden: int = 64
    depth: int = 4
    n_ff: int = 128          # nombre de fréquences de Fourier en entrée
    ff_mode: str = 'gauss'   # tirage des fréquences de Fourier, cf. WavePINN._fourier_freqs
    linear_skip: bool = False  # chemin linéaire encodage → sortie (cf. FourierMLP)
    ff_sigma: float = 1.0    # gauss : écart-type (cycles/λp, /Tp) ; dispersion : écart relatif
    iters: int = 4000
    lr: float = 1e-3
    n_colloc: int = 4096     # points de collocation tirés à chaque itération
    horizon: float = 90.0    # s de prédiction après la fenêtre
    x_max: float = 100.0     # m : aval du domaine (navire en x = 0)
    x_margin: float = 50.0   # m : marge en amont de la sonde la plus lointaine
    w_data: float = 1.0
    w_phys: float = 1.0
    phys_ramp: int = 500     # itérations : poids physique monté de 0 à w_phys (données d'abord)
    adapt_every: int = 0     # > 0 : w_phys réajusté (normes de gradients) toutes les N itérations
    causal_eps: float = 0.0  # > 0 : causalité spatiale, ε final (annelé depuis ε/100 ; cf. _phys_loss)
    causal_chunks: int = 16  # tranches en x pour la causalité
    lbfgs_iters: int = 0     # itérations L-BFGS après Adam (0 = aucune)
    recon_frac: float = 0.99  # bande d'énergie qui fixe λ_max (donc la profondeur h)
    seed: int = 0
    device: str = 'auto'
    log_every: int = 0


def config_from_kv(pairs, **defaults):
    """PINNConfig depuis des chaînes 'clé=valeur' (ligne de commande)."""
    types = {f.name: f.type for f in dataclasses.fields(PINNConfig)}
    over = dict(defaults)
    for k, v in (p.split('=', 1) for p in pairs):
        if k not in types:
            raise KeyError(f'paramètre PINN inconnu : {k}')
        over[k] = (v.lower() in ('1', 'true')) if types[k] is bool else types[k](v)
    return PINNConfig(**over)


class FourierMLP(nn.Module):
    """
    MLP tanh précédé d'un encodage de Fourier [sin, cos](2π(bₓx + bₜt)).
    Avec z : chaque encodage est multiplié par e^{2π|bₓ|z}, ce qui le rend
    harmonique en (x, z) — la bonne décroissance verticale dès l'entrée.
    """

    def __init__(self, B, hidden, depth, linear_skip=False):
        super().__init__()
        self.register_buffer('B', B)                     # (2, n_ff)
        dims = [2 * B.shape[1]] + [hidden] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, 1))
        # Chemin linéaire encodage → sortie : avec ff_mode='dispersion', il
        # représente directement une somme d'ondes d'Airy (option 2) ; le MLP
        # ne porte que les corrections.
        self.skip = nn.Linear(2 * B.shape[1], 1, bias=False) if linear_skip else None

    def forward(self, x, t, z=None):
        arg = TWO_PI * (x * self.B[0] + t * self.B[1])
        f = torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)
        if z is not None:
            f = f * torch.exp(TWO_PI * self.B[0].abs() * z).repeat(1, 2)
        return self.net(f) + (self.skip(f) if self.skip is not None else 0.0)


def _d(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residuals(eta_net, phi_net, xs, ts, xi, ti, zi, xb, tb, h):
    """
    Résidus de la théorie potentielle linéarisée (unités réduites).
    (xs, ts) : surface z = 0 ; (xi, ti, zi) : intérieur ; (xb, tb) : fond z = −h.
    Tous les tenseurs d'entrée doivent avoir requires_grad.
    Chaque résidu est divisé par son échelle au pic (ω̃p = k̃p = 2π) : erreur
    relative d'ordre 1, sinon la physique (~ω̃² ≈ 60) écrase les données et
    le réseau converge vers la solution triviale η = φ = 0.
    Retourne (r_kin, r_dyn, r_lap, r_bot).
    """
    zs = torch.zeros_like(xs, requires_grad=True)
    eta, phi = eta_net(xs, ts), phi_net(xs, ts, zs)
    r_kin = (_d(eta, ts) - _d(phi, zs)) / TWO_PI
    r_dyn = (_d(phi, ts) + TWO_PI * eta) / TWO_PI
    phi_i = phi_net(xi, ti, zi)
    r_lap = (_d(_d(phi_i, xi), xi) + _d(_d(phi_i, zi), zi)) / TWO_PI**2
    zb = torch.full_like(xb, -h, requires_grad=True)
    r_bot = _d(phi_net(xb, tb, zb), zb) / TWO_PI
    return r_kin, r_dyn, r_lap, r_bot


def _col(a, dev):
    return torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(-1, 1), device=dev)


class WavePINN:
    """Un PINN ajusté sur une fenêtre de mesures ; .predict(x, t) en mètres."""

    def __init__(self, Tp, gamma, Hs_scale, cfg=PINNConfig()):
        self.cfg = cfg
        self.dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                                if cfg.device == 'auto' else cfg.device)
        torch.manual_seed(cfg.seed)
        self.Tp, self.lam = Tp, g * Tp**2 / TWO_PI
        self.eta_scale = Hs_scale / 4.0
        w_lo, w_hi = energy_band(Tp, gamma, cfg.recon_frac)
        self.h = 0.5 * (TWO_PI / Tp / w_lo) ** 2        # λ_max / 2, en unités de λp
        self.B = B = self._fourier_freqs(w_lo * Tp / TWO_PI, w_hi * Tp / TWO_PI)
        self.eta_net = FourierMLP(B, cfg.hidden, cfg.depth, cfg.linear_skip).to(self.dev)
        self.phi_net = FourierMLP(B, cfg.hidden, cfg.depth, cfg.linear_skip).to(self.dev)
        self.history = []

    def _fourier_freqs(self, f_lo, f_hi):
        """
        Fréquences (bₓ, bₜ) de l'encodage, en cycles par λp et par Tp.
        f_lo, f_hi : bande de fréquence temporelle reconstruite (en 1/Tp).
          gauss      : N(0, ff_sigma²) sur les deux axes (générique)
          box        : bₜ dans la bande, bₓ uniforme sur [0, f_hi²]
          dispersion : bₓ = bₜ² · (1 + ff_sigma·N(0,1)), autour de la
                       dispersion eau profonde (en cycles : fₓ = fₜ²)
        box et dispersion n'encodent que des ondes allant vers +x.
        """
        cfg, n = self.cfg, self.cfg.n_ff
        if cfg.ff_mode == 'gauss':
            return torch.randn(2, n) * cfg.ff_sigma
        ft = f_lo + (f_hi - f_lo) * torch.rand(n)
        if cfg.ff_mode == 'box':
            fx = f_hi**2 * torch.rand(n)
        elif cfg.ff_mode == 'dispersion':
            fx = ft**2 * (1 + cfg.ff_sigma * torch.randn(n))
        else:
            raise ValueError(f'ff_mode inconnu : {cfg.ff_mode}')
        return torch.stack([fx, -ft])

    # ── Passage mètres/secondes ↔ unités réduites ──
    def _x(self, x):
        return np.asarray(x, float) / self.lam

    def _t(self, t):
        return (np.asarray(t, float) - self.t0) / self.Tp

    def _sample(self, n):
        """Points de collocation (surface, intérieur, fond), chacun avec
        l'indice de sa tranche en x (amont → aval) pour la causalité."""
        dev, M = self.dev, max(1, self.cfg.causal_chunks)
        x_lo, x_hi, t_lo, t_hi = self._box

        def pts(m):
            c = torch.arange(m, device=dev) % M                  # tirage stratifié par tranche
            u = torch.rand(m, 2, device=dev)
            x = x_lo + (x_hi - x_lo) * (c + u[:, 0]) / M
            t = t_lo + (t_hi - t_lo) * u[:, 1]
            return x[:, None].requires_grad_(), t[:, None].requires_grad_(), c

        (xs, ts, cs), (xi, ti, ci), (xb, tb, cb) = pts(n), pts(n), pts(max(M, n // 4))
        zi = (-self.h * torch.rand_like(xi) ** 2).requires_grad_()   # dense près de z = 0
        return (xs, ts, xi, ti, zi, xb, tb), (cs, ci, cb)

    def _phys_loss(self, pts, chunks, eps=None):
        """
        Perte physique. Avec causal_eps > 0 (causalité en espace, cf. Wang,
        Sankaran & Perdikaris 2022 — ici selon x, sens de propagation) : la
        tranche c ne compte que quand les tranches en amont sont résolues,
            w_c = exp(−ε Σ_{c' < c} L_c')   (sans gradient)
        ce qui force l'information à partir des sondes vers le navire.
        eps : valeur courante de ε (annelée par fit), causal_eps par défaut.
        Retourne (perte, résidus moyens (kin, dyn, lap, fond), poids min).
        """
        M = max(1, self.cfg.causal_chunks)
        eps = self.cfg.causal_eps if eps is None else eps
        r = residuals(self.eta_net, self.phi_net, *pts, self.h)
        comps = [q.pow(2).squeeze(1) for q in r]
        means = [q.mean() for q in comps]
        if eps <= 0 or M == 1:
            return sum(means), means, 1.0
        cs, ci, cb = chunks
        per_chunk = sum(torch.zeros(M, device=self.dev).index_add_(0, c, q)
                        / torch.bincount(c, minlength=M).clamp(min=1)
                        for q, c in zip(comps, (cs, cs, ci, cb)))
        upstream = torch.cumsum(per_chunk, 0) - per_chunk
        w = torch.exp(-eps * upstream).detach()
        return (w * per_chunk).mean(), means, w.min().item()

    def fit(self, t, x_probes, eta_mes):
        """t [Nt] (fenêtre), x_probes [Np], eta_mes [Np, Nt] en mètres."""
        cfg, dev = self.cfg, self.dev
        self.t0 = t[-1] + (t[1] - t[0])
        xx, tt = np.meshgrid(x_probes, t, indexing='ij')
        xd, td = _col(self._x(xx), dev), _col(self._t(tt), dev)
        ed = _col(np.asarray(eta_mes) / self.eta_scale, dev)
        self._box = (self._x(np.min(x_probes) - cfg.x_margin), self._x(cfg.x_max),
                     self._t(t[0]), cfg.horizon / self.Tp)

        params = list(self.eta_net.parameters()) + list(self.phi_net.parameters())
        w_phys = cfg.w_phys

        def data_loss():
            return ((self.eta_net(xd, td) - ed) ** 2).mean()

        def log(it, l_data, means, w_min):
            self.history.append((it, l_data, *[m.item() for m in means], w_phys, w_min))
            if cfg.log_every and it % cfg.log_every == 0:
                print('it %5d  data %.2e  kin %.2e  dyn %.2e  lap %.2e  fond %.2e'
                      '  w_phys %.2g  causal_min %.2f' % self.history[-1], flush=True)

        opt = torch.optim.Adam(params, lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, cfg.iters))
        for it in range(cfg.iters):
            l_data = data_loss()
            # ε causal monté géométriquement de causal_eps/100 (tolérant) à causal_eps
            eps = cfg.causal_eps * 100.0 ** (it / max(1, cfg.iters - 1) - 1.0)
            l_phys, means, w_min = self._phys_loss(*self._sample(cfg.n_colloc), eps)
            if cfg.adapt_every and it >= cfg.phys_ramp and it % cfg.adapt_every == 0:
                # Normes des gradients égalisées : la physique pèse autant que les données
                gd = torch.autograd.grad(cfg.w_data * l_data, params, retain_graph=True,
                                         allow_unused=True)
                gp = torch.autograd.grad(l_phys, params, retain_graph=True, allow_unused=True)
                norm = lambda gs: torch.sqrt(sum((q**2).sum() for q in gs if q is not None))
                w_phys = 0.9 * w_phys + 0.1 * (norm(gd) / (norm(gp) + 1e-12)).item()
            ramp = min(1.0, it / cfg.phys_ramp) if cfg.phys_ramp else 1.0
            loss = cfg.w_data * l_data + ramp * w_phys * l_phys
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if it % 50 == 0 or it == cfg.iters - 1:
                log(it, l_data.item(), means, w_min)

        if cfg.lbfgs_iters:
            # L-BFGS sur un jeu de collocation fixe (perte déterministe)
            batch = self._sample(2 * cfg.n_colloc)
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.lbfgs_iters,
                                    history_size=50, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = cfg.w_data * data_loss() + w_phys * self._phys_loss(*batch)[0]
                loss.backward()
                return loss
            opt.step(closure)
            l_phys, means, w_min = self._phys_loss(*batch)
            log(cfg.iters + cfg.lbfgs_iters, data_loss().item(), means, w_min)
        self.w_phys = w_phys
        return self

    @torch.no_grad()
    def predict(self, x, t):
        """η̂ sur la grille (t, x) : [Nt, Nx] en mètres."""
        tt, xx = np.meshgrid(t, x, indexing='ij')
        xs, ts = _col(self._x(xx), self.dev), _col(self._t(tt), self.dev)
        out = torch.cat([self.eta_net(a, b) for a, b in
                         zip(xs.split(65536), ts.split(65536))])
        return out.cpu().numpy().reshape(tt.shape) * self.eta_scale


def forecaster(Tp, gamma, cfg=PINNConfig()):
    """Prédicteur au format evaluate : (t, x, η) → f(x, t) = η̂ [Nt, Nx].
    L'échelle Hs vient des mesures (4·écart-type), jamais de la vérité."""
    def run(t, x, eta_mes):
        Hs_est = 4.0 * float(np.std(eta_mes))
        return WavePINN(Tp, gamma, Hs_est, cfg).fit(t, x, eta_mes).predict
    return run
