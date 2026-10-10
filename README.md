# wellfit

Goodness-of-fit statistics for model fits to data with correlated noise and imperfect models. It works with any 1D
data (spectra, light curves, SEDs, ...). The only dependency is `numpy`.

## Why

The standard χ² with `n_data − n_params` degrees of freedom assumes independent data points. Rebinning, stacking,
interpolation or smoothing correlate neighbouring points. Then `N` points carry only `N / n_corr` independent
measurements, and naive errors are too small by about `√n_corr`. `wellfit` corrects for this, and also handles model
mismatch and data subsets that disagree with the rest.

## Install

```bash
pip install git+https://github.com/AlexanderLaroche/wellfit.git
```

## Usage

```python
import wellfit as wf

# noise, from signal-free reference ranges
err_obs_scaled, factor = wf.measure_error_scale(x, y, err_obs, ref_ranges)
n_corr, rho = wf.measure_n_corr(x, y, err_obs_scaled, ref_ranges)

# grid search: residuals (model - data) of every grid model, shape (n_models, n_data)
fit = wf.fit_grid(R, err_obs_scaled, n_corr, n_params=2, groups=groups, params={'vsini': vsini, 'vmacro': vmacro})

# or a continuous fit with any minimizer
fit = wf.fit_optimizer(residuals, minimize, theta0, err_obs_scaled, n_corr, groups=groups)

print(fit.summary())
fit.results['vsini']   # (value, +stat, -stat, +sys, -sys)
```

For MCMC, use `wf.log_likelihood(residuals, err_obs_scaled, n_corr, fit.sigma_sys)` as the likelihood.

## What it computes

| | |
|---|---|
| `measure_error_scale` | rescale the errors to the scatter in reference regions |
| `measure_n_corr` | `n_corr = 1 + 2 Σ ρ_k`, the integrated autocorrelation of the residuals |
| `chi2_eff`, `dof_eff` | `χ²_eff = χ² / n_corr`, `ν = n_data / n_corr − n_params` |
| `solve_sigma_sys` | systematic floor added in quadrature so that the best fit has `χ²_eff = ν` |
| `tension` | how much the fit to the other groups (lines, bands, epochs, ...) improves when one group is left out |
| `fit_grid`, `fit_optimizer` | all of the above, plus stat errors (posterior, curvature or profile scan) and sys errors (stratified bootstrap over groups) |

## Examples

Fitting rotational broadening to four synthetic lines, one of them inconsistent with the rest:

- [`grid_example.ipynb`](examples/grid_example.ipynb): grid search
- [`optimizer_example.ipynb`](examples/optimizer_example.ipynb): minimizer, with profile-scan errors
- [`mcmc_example.ipynb`](examples/mcmc_example.ipynb): MCMC, including thermal broadening and its degeneracy with macroturbulence

The examples also need `scipy` and `matplotlib`, and the MCMC one needs `emcee` and `corner`.

## License

MIT
