import numpy as np
import wellfit as wf

def correlated_noise(n, s, rng):
    # white noise smoothed with a Gaussian of s samples, renormalized to unit variance
    k = np.exp(-0.5 * (np.arange(-5 * s, 5 * s + 1) / s)**2)
    return np.convolve(rng.normal(size=n + k.size), k / np.sqrt(np.sum(k**2)), mode='valid')[:n]

def test_n_corr_white_and_smoothed():
    rng = np.random.default_rng(1)
    n_white, _ = wf.n_corr_from_residuals(rng.normal(size=20000))
    assert abs(n_white - 1) < 0.1
    # Gaussian smoothing over s samples: rho_k = exp(-k^2 / 4s^2), n_corr ~ sqrt(4 pi) s
    s = 1.5
    n_s, _ = wf.n_corr_from_residuals(correlated_noise(20000, s, rng))
    assert abs(n_s - np.sqrt(4 * np.pi) * s) < 0.4

def test_measure_on_baseline_ranges():
    # sloped baseline + signal outside the ranges; noise 2x the quoted errors
    rng = np.random.default_rng(2)
    x = np.linspace(0., 100., 5000)
    y = 3 + 0.05 * x + 5 * np.exp(-0.5 * ((x - 50) / 2)**2) + 0.02 * rng.normal(size=x.size)
    err = np.full(x.size, 0.01)
    ranges = [(0, 40), (60, 100)]
    e2, f = wf.measure_error_scale(x, y, err, ranges)
    assert abs(f - 2) < 0.05
    n, _ = wf.measure_n_corr(x, y, e2, ranges)
    assert abs(n - 1) < 0.1

def test_sigma_sys_reaches_nu():
    rng = np.random.default_rng(3)
    R = 30. * rng.normal(size=(20, 500))  # units independent: errors of 10
    err = np.full(500, 10.)
    sig = wf.solve_sigma_sys(R, err, n_corr=2., n_params=2)
    assert np.isclose(wf.chi2_eff(R, err, 2., sig).min(), wf.dof_eff(500, 2, 2.), rtol=1e-6)
    assert wf.solve_sigma_sys(0.01 * R, err, 2., 2) == 0.

def test_tension_flags_a_discrepant_group():
    # 1-parameter grid of amplitudes; group 3 is generated with a different amplitude than groups 0-2
    rng = np.random.default_rng(4)
    t = np.linspace(-3, 3, 60)
    shape = np.exp(-0.5 * t**2)
    amps = np.linspace(0.1, 0.9, 81)
    truth = [0.5, 0.5, 0.5, 0.8]
    data = np.concatenate([a * shape for a in truth]) + 0.01 * rng.normal(size=4 * t.size)
    models = np.stack([np.tile(a * shape, 4) for a in amps])
    g = wf.fit_grid(models - data, 0.01, n_corr=1., n_params=1, groups=np.repeat(np.arange(4), t.size),
                    params={'amp': amps}, n_boot=200)
    assert np.argmax(g.tension) == 3
    assert np.isclose(amps[g.loo_best[3]], 0.5, atol=0.02)
    assert g.sigma_sys > 0
    g.summary()

def test_no_groups():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 1, 200)
    slopes = np.linspace(0, 2, 41)
    data = 1.0 * t + 0.05 * rng.normal(size=t.size)
    g = wf.fit_grid(slopes[:, None] * t - data, 0.05, n_corr=1., n_params=1, params={'slope': slopes})
    v, ep, em, sp, sm = g.results['slope']
    assert abs(v - 1) <= 0.05 and ep >= 0.05 and em >= 0.05
