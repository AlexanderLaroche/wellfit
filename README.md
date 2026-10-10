# wellfit

Goodness-of-fit statistics for fitting models to data whose points are correlated, and whose models are imperfect.
Works with any data: spectra, light curves, SEDs, radial velocities, images flattened to 1D. Depends only on numpy.

## Why

The usual recipe, chi2 with `n_data - n_params` degrees of freedom, assumes independent data points. Correlation
between neighbouring points is common. Resampling, interpolation, smoothing, stacking, or an instrumental response
wider than the sampling all cause it. With correlation, `N` points carry the information of only about `N / n_corr`
independent points. Treating them as independent:

- inflates every Delta chi2 by `n_corr`, which shrinks the confidence intervals by about `sqrt(n_corr)`;
- makes the reduced chi2 and any error floor derived from it wrong.

Models are rarely perfect either. A best fit with reduced chi2 well above 1 means the formal errors are too small.

## What it does

| step | function | what |
|---|---|---|
| error rescaling | `measure_error_scale`, `error_scale_from_residuals` | scale the errors so the residuals in signal-free reference regions (or about a fit) have unit scatter |
| correlation | `measure_n_corr`, `n_corr_from_residuals` | `n_corr = 1 + 2 sum_k rho_k`, the integrated autocorrelation of the normalized residuals, summed until it first drops below zero |
| effective chi2 | `chi2_eff`, `dof_eff` | `chi2_eff = chi2 / n_corr`, `nu = n_data / n_corr - n_params` |
| systematic floor | `solve_sigma_sys` | `sigma_sys` added in quadrature to all points so the best model has `chi2_eff = nu` |
| per-group diagnostics | `group_chi2_eff`, `tension`, `leave_one_out` | chi2_eff of each group of points (a line, band, epoch, instrument, ...) vs its expectation `n_points / n_corr`; **tension** = how much the fit to all the other groups improves when this one is left out and the model refitted (~0 if consistent; large = this group pulls the fit) |
| errors | `stat_error`, `sys_error` | stat: 16-84th percentiles of the posterior `exp(-chi2_eff/2)` over a parameter grid, at least one grid step; sys: a stratified bootstrap over groups (groups resampled within strata) |
| everything | `fit_grid`, `fit_optimizer` | all of the above, for a model grid or for continuous parameters with any minimizer |

The noise functions need the data ordered along some coordinate `x`, roughly uniformly sampled. Correlation is
measured per lag.

## Install

```bash
pip install -e /path/to/wellfit
```

## Usage

### 1. Noise (both streams)

```python
import numpy as np
import wellfit as wf

# from signal-free reference ranges in x (a polynomial baseline is fitted in each) ...
ref = [(4150, 4180), (5050, 5150)]
err, factor = wf.measure_error_scale(x, y, err, ref, deg=1)
n_corr, rho = wf.measure_n_corr(x, y, err, ref, deg=1)
# ... or from any residuals you trust, e.g. of a preliminary best fit:  n_corr, rho = wf.n_corr_from_residuals(z)
```

### 2a. Grid stream: models on a parameter grid

Give the residuals of every model, using a flattened grid.

```python
R = model_values - y_fit                  # (n_models, n_data)
g = wf.fit_grid(R, err_fit, n_corr, n_params=2,
                groups=group_of_each_point,   # optional: 0..n_groups-1 per point
                params={'vsini': vsini_of_each_model, 'Teff': teff_of_each_model},
                group_names=names, strata=stratum_of_each_group)
print(g.summary())
g.results['vsini']   # (value, +stat, -stat, +sys, -sys)
```

Results:

- stat: 16-84th percentiles of the posterior `exp(-chi2_eff/2)`, at least one grid step.
- sys: the stratified bootstrap over groups.

Both are snapped to the grid.

### 2b. Optimizer stream: continuous parameters, any minimizer

```python
from scipy.optimize import minimize

def residuals(theta):                     # model - data on the fitted points, (n_data,)
    return model(x_fit, *theta) - y_fit

g = wf.fit_optimizer(residuals, lambda f, x0: minimize(f, x0, method='Nelder-Mead').x,
                     theta0=[100., 1.0], err=err_fit, n_corr=n_corr,
                     groups=group_of_each_point, param_names=['vsini', 'scale'], n_boot=100)
print(g.summary())
g.results['vsini']   # (value, +stat, -stat, +sys, -sys)
```

What replaces the grid steps:

- The fit and `sigma_sys` are alternated until both are stable.
- stat errors come from the curvature of chi2_eff at the best fit: covariance = 2 H^-1, with `step` setting the
  finite-difference steps.
- Each group left out, each stratum left out, and each of the `n_boot` bootstrap replicates is a refit started from
  the best fit. `n_boot` therefore sets most of the cost.
- The minimizer is yours, so no scipy dependency.

Set `profile={'vsini': np.arange(...)}` to get stat errors from a Δchi2_eff = 1 profile scan instead of the
curvature. At each value of the scanned parameter the others are refitted. The errors are asymmetric and correct at a
physical bound such as v_macro = 0. Use it whenever the best fit is near a bound.

For MCMC, use `wf.log_likelihood(residuals(theta), err, n_corr, sigma_sys)` in your sampler. That's
-chi2_eff/2, so the posterior widths account for the correlated data.

### Notes

`n_params` counts every fitted parameter, including nuisance parameters optimized separately for each model, such as
a shift or a normalization.

## Examples

[`examples/`](examples) has two notebooks that fit v sin i and v_macro to four synthetic lines. The data have
correlated noise, errors quoted too small, and one deliberately inconsistent line.

- [`grid_example.ipynb`](examples/grid_example.ipynb): the grid stream.
- [`optimizer_example.ipynb`](examples/optimizer_example.ipynb): the optimizer stream, with profile scans.

Both compare the naive fit with wellfit. They need scipy and matplotlib.

## Tests

```bash
python -m pytest tests
```

Method: the error rescaling follows Gotberg et al. (2023, ApJ 959, 125).
