"""wellfit: honest goodness-of-fit statistics for model fits to correlated data.

Two streams sharing the same statistics:
    fit_grid       models evaluated on a parameter grid (residual matrix of every model)
    fit_optimizer  continuous parameters fitted with any minimizer (refits instead of argmin over the grid)
"""

from .noise import (baseline_residuals, error_scale_from_residuals, n_corr_from_residuals, measure_error_scale,
                    measure_n_corr)
from .stats import chi2_eff, dof_eff, solve_sigma_sys, group_chi2_eff, posterior
from .groups import leave_one_out, tension, bootstrap_weights, stratum_drop_weights, refit_weighted
from .grid import grid_steps, snap, weighted_quantiles, stat_error, sys_error
from .grid_fit import fit_grid, GridFit
from .opt_fit import fit_optimizer, OptFit, log_likelihood, weighted_chi2_eff, hessian

__version__ = "0.2.0"
