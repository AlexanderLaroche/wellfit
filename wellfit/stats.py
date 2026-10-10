import numpy as np

# Effective chi2 of a set of models, counted in independent data points, and the systematic error floor.
#
# R is the residual matrix model - data, shape (n_models, n_data) (or (n_data,) for one model), err the errors
# (n_data,). All chi2 here are "effective": sum(R^2 / (err^2 + sigma_sys^2)) / n_corr, so that Delta chi2_eff has its
# usual meaning (Delta chi2_eff = 1 -> 1 sigma for one parameter) even though neighboring points are correlated.

def chi2_eff(R, err, n_corr=1., sigma_sys=0.):
    """chi2 / n_corr with sigma_sys added in quadrature to the errors. R: (..., n_data). Returns (...)."""
    e2 = np.asarray(err, dtype=float)**2 + sigma_sys**2
    return np.sum(np.asarray(R)**2 / e2, axis=-1) / n_corr

def dof_eff(n_data, n_params, n_corr=1.):
    """Degrees of freedom counted in independent data points: n_data / n_corr - n_params."""
    return n_data / n_corr - n_params

def solve_sigma_sys(R, err, n_corr, n_params, hi=None, n_iter=60):
    """
    Systematic error floor sigma_sys (units of err), added in quadrature to all points, such that the best model (the
    minimum over the rows of R, re-chosen at every sigma_sys) has chi2_eff = dof_eff. 0 if the best model already has
    chi2_eff <= dof_eff. Bisection; hi is the starting upper bound (default: the median error), doubled until it brackets.
    """
    R = np.atleast_2d(R)
    err = np.asarray(err, dtype=float)
    nu = dof_eff(R.shape[1], n_params, n_corr)
    if nu <= 0:
        raise ValueError(f"no degrees of freedom left: n_data / n_corr - n_params = {nu:.2f}")
    f = lambda s: chi2_eff(R, err, n_corr, s).min()
    if f(0.) <= nu:
        return 0.
    lo, hi = 0., (np.median(err) if hi is None else hi)
    while f(hi) > nu:
        lo, hi = hi, 2 * hi
    for _ in range(n_iter):
        s = 0.5 * (lo + hi)
        lo, hi = (s, hi) if f(s) > nu else (lo, s)
    return 0.5 * (lo + hi)

def group_chi2_eff(R, err, groups, n_corr=1., sigma_sys=0., n_groups=None):
    """
    chi2_eff of each group of data points (spectral lines, bands, epochs, instruments, ...): groups (n_data,) gives
    each point's group index 0..n_groups-1 (negative = in no group, ignored). Returns (n_models, n_groups); the rows
    sum to chi2_eff when every point is in a group.
    """
    R = np.atleast_2d(R)
    groups = np.asarray(groups)
    n_groups = int(groups.max()) + 1 if n_groups is None else n_groups
    W = np.zeros((groups.size, n_groups))
    ok = groups >= 0
    W[np.flatnonzero(ok), groups[ok]] = 1. / (np.asarray(err, dtype=float)[ok]**2 + sigma_sys**2)
    return (R**2 @ W) / n_corr

def posterior(c2e):
    """Posterior over the models with flat priors on the grid axes: exp(-chi2_eff / 2), normalized."""
    c2e = np.asarray(c2e, dtype=float)
    p = np.exp(-0.5 * (c2e - c2e.min()))
    return p / p.sum()
