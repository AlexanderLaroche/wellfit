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

# --- optimizer stream (needs scipy for the minimizer only)

def _nm(f, x0):
    from scipy.optimize import minimize
    return minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 20000}).x

def test_optimizer_matches_linear_least_squares():
    import pytest; pytest.importorskip('scipy')
    rng = np.random.default_rng(6)
    t = np.linspace(0, 1, 300)
    y = 2. * t + 0.5 + 0.05 * rng.normal(size=t.size)
    err = np.full(t.size, 0.05)
    for n_corr in (1., 3.):
        g = wf.fit_optimizer(lambda th: th[0] * t + th[1] - y, _nm, [1., 0.], err, n_corr, fit_sigma_sys=False,
                             param_names=['a', 'b'])
        A = np.stack([t, np.ones_like(t)], axis=1) / 0.05
        cov = np.linalg.inv(A.T @ A) * n_corr  # least-squares covariance, inflated by n_corr
        assert np.allclose(g.theta, np.linalg.lstsq(A, y / 0.05, rcond=None)[0], atol=1e-5)
        assert np.allclose(np.sqrt(np.diag(g.cov)), np.sqrt(np.diag(cov)), rtol=0.02)

def test_optimizer_agrees_with_grid_stream():
    import pytest; pytest.importorskip('scipy')
    rng = np.random.default_rng(4)
    t = np.linspace(-3, 3, 60)
    shape = np.exp(-0.5 * t**2)
    truth = [0.5, 0.5, 0.5, 0.8]
    data = np.concatenate([a * shape for a in truth]) + 0.01 * rng.normal(size=4 * t.size)
    groups = np.repeat(np.arange(4), t.size)
    amps = np.linspace(0.1, 0.9, 801)
    gg = wf.fit_grid(np.stack([np.tile(a * shape, 4) for a in amps]) - data, 0.01, 1., 1, groups=groups,
                     params={'amp': amps}, n_boot=200)
    go = wf.fit_optimizer(lambda th: np.tile(th[0] * shape, 4) - data, _nm, [0.3], 0.01, 1., groups=groups,
                          param_names=['amp'], n_boot=50)
    assert np.isclose(go.sigma_sys, gg.sigma_sys, rtol=0.02)
    assert np.isclose(go.theta[0], gg.results['amp'][0], atol=0.002)
    assert np.allclose(go.tension, gg.tension, rtol=0.05, atol=0.5)
    assert np.argmax(go.tension) == 3 and np.isclose(go.theta_loo[3, 0], 0.5, atol=0.01)
    go.summary()

def test_log_likelihood_is_minus_half_chi2_eff():
    R = np.array([1., -2., 0.5]); err = np.array([1., 1., 0.5])
    ll0 = wf.log_likelihood(np.zeros(3), err, 2.)
    assert np.isclose(wf.log_likelihood(R, err, 2.) - ll0, -0.5 * wf.chi2_eff(R, err, 2.))

def test_profile_errors_match_curvature_and_handle_a_bound():
    import pytest; pytest.importorskip('scipy')
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 300)
    y = 2. * t + 0.5 + 0.05 * rng.normal(size=t.size)
    g = wf.fit_optimizer(lambda th: th[0] * t + th[1] - y, _nm, [1., 0.], 0.05, 1., fit_sigma_sys=False,
                         param_names=['a', 'b'], profile={'a': np.linspace(1.6, 2.4, 161)})
    _, ep, em, _, _ = g.results['a']
    assert np.isclose(ep, np.sqrt(g.cov[0, 0]), rtol=0.02) and np.isclose(em, np.sqrt(g.cov[0, 0]), rtol=0.02)
    # best value at a bound: the lower error reaches the bound, the upper one is the Delta chi2_eff = 1 crossing
    v = np.linspace(0, 1, 101); prof = (v / 0.2)**2
    assert wf.profile_errors(v, prof, 0.)[1] == 0. and np.isclose(wf.profile_errors(v, prof, 0.)[0], 0.2, atol=0.002)
