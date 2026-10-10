"""wellfit: honest goodness-of-fit statistics for model fits to correlated data."""

from .noise import (baseline_residuals, error_scale_from_residuals, n_corr_from_residuals, measure_error_scale,
                    measure_n_corr)
from .stats import chi2_eff, dof_eff, solve_sigma_sys, group_chi2_eff, posterior
from .groups import leave_one_out, tension, bootstrap_weights, stratum_drop_weights, refit_weighted
from .grid import grid_steps, snap, weighted_quantiles, stat_error, sys_error
from .fit import fit_grid, GridFit

__version__ = "0.1.0"
