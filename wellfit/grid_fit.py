from dataclasses import dataclass, field
import numpy as np

from .stats import chi2_eff, dof_eff, solve_sigma_sys, group_chi2_eff, posterior
from .groups import leave_one_out, tension, bootstrap_weights, stratum_drop_weights, refit_weighted
from .grid import stat_error, sys_error

@dataclass
class GridFit:
    n_corr: float
    n_params: int
    n_data: int
    nu: float                    # degrees of freedom in independent points, n_data / n_corr - n_params
    sigma_sys: float             # error floor giving the best model chi2_eff = nu
    chi2: np.ndarray             # raw chi2 of every model (given errors only, n_corr = 1)
    chi2_eff: np.ndarray         # (chi2 with sigma_sys) / n_corr
    post: np.ndarray             # posterior weights over the models
    best: int
    group_names: list
    strata: np.ndarray
    chi2_eff_groups: np.ndarray  # (n_models, n_groups)
    n_data_groups: np.ndarray
    tension: np.ndarray
    loo_best: np.ndarray         # best model leaving out each group
    boot_best: np.ndarray        # best model in each bootstrap replicate
    stratum_best: dict           # best model leaving out each stratum
    params: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)  # name -> (value, +stat, -stat, +sys, -sys)

    def summary(self):
        s = [f"n_data = {self.n_data}, n_corr = {self.n_corr:.2f} -> {self.n_data / self.n_corr:.1f} independent "
             f"points, nu = {self.nu:.1f}",
             f"sigma_sys = {self.sigma_sys:.4g}; best model {self.best}: chi2 = {self.chi2[self.best]:.1f} (raw), "
             f"chi2_eff = {self.chi2_eff[self.best]:.1f}"]
        if self.results:
            s.append("Parameters (value +stat -stat, +sys -sys):")
            for k, (v, ep, em, sp, sm) in self.results.items():
                s.append(f"  {k:10s} {v:10.4g}  +{ep:.4g} -{em:.4g}   sys +{sp:.4g} -{sm:.4g}")
        if len(self.group_names) > 1:
            s.append("Groups (chi2_eff / expected, tension" +
                     (", leave-one-out best " + ", ".join(self.params) if self.params else "") + "):")
            c2 = self.chi2_eff_groups[self.best]
            for j, n in enumerate(self.group_names):
                loo = "  ".join(f"{a[self.loo_best[j]]:.4g}" for a in self.params.values())
                s.append(f"  {n:16s} {str(self.strata[j]):8s} {c2[j]:8.1f} / {self.n_data_groups[j] / self.n_corr:6.1f}"
                         f"  tension {self.tension[j]:6.1f}  {loo}")
        return "\n".join(s)

def fit_grid(R, err, n_corr, n_params, groups=None, params=None, group_names=None, strata=None, n_boot=1000, seed=0,
             fit_sigma_sys=True):
    """
    Full analysis from the residuals R = model - data (n_models, n_data) of a set of models (typically a parameter grid,
    flattened).

    err: data errors (n_data,), ideally rescaled first (wellfit.measure_error_scale / error_scale_from_residuals).
    n_corr: correlation factor (wellfit.measure_n_corr / n_corr_from_residuals); 1 for independent points.
    n_params: number of fitted parameters, including nuisance parameters optimized separately for each model.
    groups: group index of every data point (n_data,), 0..n_groups-1; default: all points one group (the per-group
        diagnostics and the bootstrap then do nothing useful, and the sys errors are 0).
    params: {name: values over the models (n_models,)} for the parameters to report errors on.
    strata: stratum label of every group, for the stratified bootstrap (default: one stratum).
    fit_sigma_sys: add the error floor that brings the best model to chi2_eff = nu (else sigma_sys = 0).
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    err = np.broadcast_to(np.asarray(err, dtype=float), R.shape[1:])
    n_models, n_data = R.shape
    groups = np.zeros(n_data, dtype=int) if groups is None else np.asarray(groups)
    n_groups = int(groups.max()) + 1
    group_names = [f"group{j}" for j in range(n_groups)] if group_names is None else list(group_names)
    strata = np.zeros(n_groups, dtype=int) if strata is None else np.asarray(strata)
    params = {} if params is None else {k: np.asarray(v, dtype=float) for k, v in params.items()}

    sig = solve_sigma_sys(R, err, n_corr, n_params) if fit_sigma_sys else 0.
    C = group_chi2_eff(R, err, groups, n_corr, sig, n_groups)
    c2e = chi2_eff(R, err, n_corr, sig)
    post = posterior(c2e)
    best = int(np.argmax(post))

    boot_best = refit_weighted(C, bootstrap_weights(strata, n_boot, seed))
    Ws, s_names = stratum_drop_weights(strata)
    stratum_best = dict(zip(s_names, refit_weighted(C, Ws))) if len(s_names) > 1 else {}

    results = {}
    for k, a in params.items():
        v, ep, em = stat_error(a, post, best)
        results[k] = (v, ep, em, *sys_error(a[boot_best], v, a))

    return GridFit(n_corr=n_corr, n_params=n_params, n_data=n_data, nu=dof_eff(n_data, n_params, n_corr),
                   sigma_sys=sig, chi2=chi2_eff(R, err), chi2_eff=c2e, post=post, best=best, group_names=group_names,
                   strata=strata, chi2_eff_groups=C, n_data_groups=np.bincount(groups[groups >= 0], minlength=n_groups),
                   tension=tension(C, best), loo_best=leave_one_out(C), boot_best=boot_best, stratum_best=stratum_best,
                   params=params, results=results)
