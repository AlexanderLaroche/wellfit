from dataclasses import dataclass, field
import numpy as np

from .stats import chi2_eff, dof_eff, solve_sigma_sys, group_chi2_eff
from .groups import bootstrap_weights, stratum_drop_weights

# Optimizer stream: the model is evaluated at continuous parameters theta (not pinned to a grid) and fitted with any
# minimizer. Every quantity of the grid stream has an analogue here, with the "argmin over the grid" replaced by a
# refit:
#   sigma_sys      -> alternate fitting and solving for sigma_sys until both are stable
#   stat errors    -> curvature of chi2_eff at the best fit (covariance = 2 H^-1), or a Delta chi2_eff = 1 profile scan
#   tension, LOO   -> one refit per group left out
#   sys errors     -> one refit per stratified bootstrap replicate over groups
# Data points in no group (groups < 0) are always included.

def log_likelihood(R, err, n_corr=1., sigma_sys=0.):
    """
    Gaussian log-likelihood counted in independent points, for MCMC: -chi2_eff / 2 - (1 / 2n_corr) sum log(err^2 + s^2)
    (the second term matters only when sigma_sys is sampled as a free parameter). R: (..., n_data).
    """
    e2 = np.asarray(err, dtype=float)**2 + sigma_sys**2
    return -0.5 * chi2_eff(R, err, n_corr, sigma_sys) - 0.5 * np.sum(np.log(2 * np.pi * e2)) / n_corr

def weighted_chi2_eff(R, err, n_corr, sigma_sys, groups, w):
    """chi2_eff with each point weighted by the weight w of its group (points with groups < 0: weight 1)."""
    pw = np.where(groups >= 0, np.asarray(w, dtype=float)[np.maximum(groups, 0)], 1.)
    return np.sum(pw * np.asarray(R)**2 / (np.asarray(err, dtype=float)**2 + sigma_sys**2)) / n_corr

def hessian(f, x, step):
    """Central finite-difference Hessian of a scalar function f at x, with steps step (one per parameter)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.empty((n, n))
    f0 = f(x)
    E = np.diag(step)
    for i in range(n):
        H[i, i] = (f(x + E[i]) - 2 * f0 + f(x - E[i])) / step[i]**2
        for j in range(i):
            H[i, j] = H[j, i] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) \
                                / (4 * step[i] * step[j])
    return H

def profile_scan(residuals, minimize, theta, i, values, err, n_corr, sigma_sys=0.):
    """
    Profile of chi2_eff along parameter i: at each value, parameter i is fixed and the others refitted (started from
    theta). Returns (chi2_eff (n_values,), theta at each value (n_values, n_params)).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    others = np.flatnonzero(np.arange(theta.size) != i)
    prof, thetas = np.empty(len(values)), np.empty((len(values), theta.size))
    start = theta[others]
    for k, v in enumerate(values):
        def full(sub):
            th = theta.copy(); th[i] = v; th[others] = sub
            return th
        if others.size:
            start = np.atleast_1d(np.asarray(minimize(lambda sub: chi2_eff(residuals(full(sub)), err, n_corr, sigma_sys),
                                                      start), dtype=float))
        thetas[k] = full(start)
        prof[k] = chi2_eff(residuals(thetas[k]), err, n_corr, sigma_sys)
    return prof, thetas

def profile_errors(values, prof, best_val, best_chi2=None, delta=1.):
    """
    Errors from a chi2_eff profile: where Delta chi2_eff crosses delta on each side of best_val (interpolated in
    sqrt(Delta chi2_eff)), relative to the lower of best_chi2 and the profile minimum. Where the profile does not cross before the end of the
    scan (e.g. a physical bound such as 0), the error reaches the scan edge, so a lower error equal to best_val -
    values[0] at a bound of 0 means "consistent with 0". Returns (+err, -err).
    """
    values, prof = np.asarray(values, dtype=float), np.asarray(prof, dtype=float)
    # interpolate sqrt(Delta chi2_eff), linear in the parameter for a parabolic profile
    d = np.sqrt(np.maximum(prof - (prof.min() if best_chi2 is None else min(best_chi2, prof.min())), 0.)) - np.sqrt(delta)
    def cross(idx):  # idx: profile indices walking away from the best value
        for a, b in zip(idx[:-1], idx[1:]):
            if d[a] <= 0 < d[b]:
                return values[a] + (values[b] - values[a]) * (-d[a]) / (d[b] - d[a])
        return values[idx[-1]]
    up = np.flatnonzero(values >= best_val); dn = np.flatnonzero(values <= best_val)[::-1]
    hi = cross(up) if up.size > 1 else values[-1]
    lo = cross(dn) if dn.size > 1 else values[0]
    return max(hi - best_val, 0.), max(best_val - lo, 0.)

@dataclass
class OptFit:
    n_corr: float
    n_params: int
    n_data: int
    nu: float                    # degrees of freedom in independent points, n_data / n_corr - n_params
    sigma_sys: float             # error floor giving the best fit chi2_eff = nu
    theta: np.ndarray            # best-fit parameters
    chi2: float                  # raw chi2 of the best fit (given errors only, n_corr = 1)
    chi2_eff: float              # (chi2 with sigma_sys) / n_corr of the best fit
    cov: np.ndarray              # stat covariance, 2 H^-1 of chi2_eff (NaN if H is not positive definite)
    param_names: list
    group_names: list
    strata: np.ndarray
    chi2_eff_groups: np.ndarray  # (n_groups,) at the best fit
    n_data_groups: np.ndarray
    tension: np.ndarray
    theta_loo: np.ndarray        # (n_groups, n_params) best fit leaving out each group
    theta_boot: np.ndarray       # (n_boot, n_params) best fit in each bootstrap replicate
    theta_stratum: dict          # best fit leaving out each stratum
    results: dict = field(default_factory=dict)  # name -> (value, +stat, -stat, +sys, -sys)
    profiles: dict = field(default_factory=dict) # name -> (values, chi2_eff profile) for the profiled parameters

    def summary(self):
        s = [f"n_data = {self.n_data}, n_corr = {self.n_corr:.2f} -> {self.n_data / self.n_corr:.1f} independent "
             f"points, nu = {self.nu:.1f}",
             f"sigma_sys = {self.sigma_sys:.4g}; best fit: chi2 = {self.chi2:.1f} (raw), chi2_eff = {self.chi2_eff:.1f}",
             "Parameters (value +stat -stat, +sys -sys):"]
        for k, (v, ep, em, sp, sm) in self.results.items():
            s.append(f"  {k:10s} {v:10.4g}  +{ep:.4g} -{em:.4g}   sys +{sp:.4g} -{sm:.4g}")
        if len(self.group_names) > 1:
            s.append("Groups (chi2_eff / expected, tension, leave-one-out best " + ", ".join(self.param_names) + "):")
            for j, n in enumerate(self.group_names):
                loo = "  ".join(f"{v:.4g}" for v in self.theta_loo[j])
                s.append(f"  {n:16s} {str(self.strata[j]):8s} {self.chi2_eff_groups[j]:8.1f} / "
                         f"{self.n_data_groups[j] / self.n_corr:6.1f}  tension {self.tension[j]:6.1f}  {loo}")
        return "\n".join(s)

def fit_optimizer(residuals, minimize, theta0, err, n_corr, n_params=None, groups=None, param_names=None,
                  group_names=None, strata=None, n_boot=100, seed=0, fit_sigma_sys=True, step=None, max_iter=20,
                  rtol_sys=1e-3, profile=None):
    """
    Full analysis of a fit with continuous parameters.

    residuals(theta) -> model - data on the fitted points (n_data,).
    minimize(objective, theta_start) -> best theta: any minimizer, e.g.
        lambda f, x0: scipy.optimize.minimize(f, x0, method='Nelder-Mead').x
        It is called once per sigma_sys iteration, group left out, stratum left out and bootstrap replicate (all
        starting from the best fit), so n_boot sets most of the cost. Profile scans call it with the scanned
        parameter removed (one fewer dimension), so anything per-parameter in it (bounds) must follow len(theta_start).
    err: data errors (n_data,); n_corr: correlation factor (wellfit.measure_n_corr / n_corr_from_residuals).
    n_params: number of fitted parameters (default len(theta0)); include nuisance parameters fitted inside residuals.
    groups: group index of every point (n_data,), 0..n_groups-1 (default: one group; no tension or sys errors).
    strata: stratum label of every group, for the stratified bootstrap (default: one stratum).
    step: finite-difference steps for the Hessian (default 1e-3 * max(|theta|, 1)); should be well below the errors
        but large enough to beat any numerical noise in residuals.
    profile: {name: values} -> stat errors of these parameters from a chi2_eff profile scan (Delta chi2_eff = 1, the
        other parameters refitted at each value) instead of the curvature: asymmetric, and correct at a physical bound
        (start the scan at the bound). Use it whenever the best fit is near a bound or the curvature errors look odd.
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    err = np.asarray(err, dtype=float)
    R0 = np.asarray(residuals(theta0), dtype=float)
    n_data = R0.size
    err = np.broadcast_to(err, (n_data,))
    n_params = theta0.size if n_params is None else n_params
    groups = np.zeros(n_data, dtype=int) if groups is None else np.asarray(groups)
    n_groups = int(groups.max()) + 1
    param_names = [f"p{i}" for i in range(theta0.size)] if param_names is None else list(param_names)
    group_names = [f"group{j}" for j in range(n_groups)] if group_names is None else list(group_names)
    strata = np.zeros(n_groups, dtype=int) if strata is None else np.asarray(strata)

    def refit(w, sig, start):
        return np.atleast_1d(np.asarray(
            minimize(lambda th: weighted_chi2_eff(residuals(th), err, n_corr, sig, groups, w), start), dtype=float))

    # best fit and sigma_sys, alternated until stable
    ones = np.ones(n_groups)
    sig = 0.
    theta = refit(ones, sig, theta0)
    for _ in range(max_iter if fit_sigma_sys else 0):
        new = solve_sigma_sys(residuals(theta), err, n_corr, n_params)
        if abs(new - sig) <= rtol_sys * max(new, 1e-300):
            sig = new
            break
        sig = new
        theta = refit(ones, sig, theta)
    R = np.asarray(residuals(theta), dtype=float)

    # stat: curvature of chi2_eff
    step = 1e-3 * np.maximum(np.abs(theta), 1.) if step is None else np.broadcast_to(np.asarray(step, float), theta.shape)
    H = hessian(lambda th: chi2_eff(residuals(th), err, n_corr, sig), theta, step)
    try:
        np.linalg.cholesky(H)
        cov = 2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full_like(H, np.nan)

    # groups: leave-one-out refits and tension = drop of the other groups' chi2_eff when this group is left out
    C = group_chi2_eff(R, err, groups, n_corr, sig, n_groups)[0]
    if n_groups > 1:
        theta_loo = np.array([refit(ones - np.eye(n_groups)[j], sig, theta) for j in range(n_groups)])
        tension = np.empty(n_groups)
        for j in range(n_groups):
            Cl = group_chi2_eff(residuals(theta_loo[j]), err, groups, n_corr, sig, n_groups)[0]
            tension[j] = (C.sum() - C[j]) - (Cl.sum() - Cl[j])
        Wb = bootstrap_weights(strata, n_boot, seed)
        theta_boot = np.array([refit(Wb[:, b], sig, theta) for b in range(n_boot)])
        Ws, s_names = stratum_drop_weights(strata)
        theta_stratum = {n: refit(Ws[:, i], sig, theta) for i, n in enumerate(s_names)} if len(s_names) > 1 else {}
    else:
        theta_loo, tension = np.full((1, theta.size), np.nan), np.zeros(1)
        theta_boot, theta_stratum = np.empty((0, theta.size)), {}

    profiles = {}
    for k, vals in (profile or {}).items():
        i = param_names.index(k)
        profiles[k] = (np.asarray(vals, dtype=float),
                       profile_scan(residuals, minimize, theta, i, vals, err, n_corr, sig)[0])

    results = {}
    c2e_best = float(chi2_eff(R, err, n_corr, sig))
    for i, k in enumerate(param_names):
        ep = em = np.sqrt(cov[i, i])
        if k in profiles:
            ep, em = profile_errors(*profiles[k], theta[i], c2e_best)
        if theta_boot.shape[0]:
            q16, q84 = np.percentile(theta_boot[:, i], [16, 84])
            sp, sm = max(q84 - theta[i], 0.), max(theta[i] - q16, 0.)
        else:
            sp = sm = 0.
        results[k] = (theta[i], ep, em, sp, sm)

    return OptFit(n_corr=n_corr, n_params=n_params, n_data=n_data, nu=dof_eff(n_data, n_params, n_corr), sigma_sys=sig,
                  theta=theta, chi2=float(chi2_eff(R, err)), chi2_eff=float(chi2_eff(R, err, n_corr, sig)), cov=cov,
                  param_names=param_names, group_names=group_names, strata=strata, chi2_eff_groups=C,
                  n_data_groups=np.bincount(groups[groups >= 0], minlength=n_groups), tension=tension,
                  theta_loo=theta_loo, theta_boot=theta_boot, theta_stratum=theta_stratum, results=results, profiles=profiles)
