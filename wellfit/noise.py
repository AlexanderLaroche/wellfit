import numpy as np

# Noise properties of ordered data y(x) with errors err: error rescaling and the correlation factor n_corr.
#
# Both are measured from normalized residuals z = (data - reference) / err in segments where the reference is known:
#   - measure_* : the reference is a low-order polynomial baseline fitted in each of the given x-ranges (regions with
#     no signal, e.g. spectral continuum, off-pulse phases, a flat part of a light curve), or
#   - *_from_residuals : any residuals you supply, e.g. data - best-fit model.
# The samples within a segment must be ordered in x and (roughly) uniformly spaced: correlation is measured per lag.
# Error rescaling follows Gotberg et al. (2023, ApJ 959, 125; https://doi.org/10.5281/zenodo.8075952).

def baseline_residuals(x, y, err, ranges=None, deg=1, relative=False):
    """
    Normalized residuals about a polynomial baseline of degree deg fitted in each x-range (ranges=None: all the data
    as one segment). relative=False: (y - fit) / err; relative=True: (y / fit - 1) / err (data normalized to a level,
    e.g. continuum-normalized spectra). Returns a list of arrays, one per segment.
    """
    x, y, err = (np.asarray(a, dtype=float) for a in (x, y, err))
    ranges = [(x.min() - 1., x.max() + 1.)] if ranges is None else ranges
    out = []
    for lo, hi in ranges:
        m = (x > lo) & (x < hi)
        if m.sum() < deg + 2:
            raise ValueError(f"range ({lo}, {hi}) has only {m.sum()} points for a degree-{deg} baseline")
        fit = np.polyval(np.polyfit(x[m], y[m], deg), x[m])
        out.append(((y[m] / fit - 1.) if relative else (y[m] - fit)) / err[m])
    return out

def _segments(z):
    return [np.asarray(z, dtype=float)] if np.ndim(z[0]) == 0 else [np.asarray(s, dtype=float) for s in z]

def error_scale_from_residuals(z, step=0.01):
    """
    Factor by which the errors must be multiplied for the normalized residuals z (one array, or a list of segments) to
    have unit standard deviation, stepped by `step` (0.01 as in Gotberg+23; step=None gives the exact std).
    """
    z = np.concatenate(_segments(z))
    std = np.std(z)  # maximum-likelihood Gaussian width
    if step is None:
        return std
    factor = 1.
    if std > 1.:
        while np.std(z / factor) > 1.:
            factor += step
    elif std < 1.:
        while np.std(z / factor) < 1.:
            factor -= step
    return factor

def n_corr_from_residuals(z, kmax=15):
    """
    Correlation factor n_corr = 1 + 2 sum_k rho_k: the integrated autocorrelation of the residuals z (one array, or a
    list of segments, each ordered in x), summed until rho_k first drops to or below zero (or k = kmax).
    A sum of squared residuals over N correlated points carries the information of N / n_corr independent points, so
    chi2 and the number of data points are both divided by n_corr (see wellfit.stats). Returns (n_corr, rho_1..kmax).

    n_corr = 1 for independent points. Any resampling, smoothing, interpolation, stacking, or an instrumental response
    wider than the sampling correlates neighbors (n_corr > 1); the naive chi2 with n_data - n_params degrees of freedom
    then over-weights the data and underestimates the errors by ~sqrt(n_corr).
    """
    num, den = np.zeros(kmax + 1), 0.
    for s in _segments(z):
        s = s - s.mean()
        den += np.sum(s * s)
        for k in range(1, min(kmax, s.size - 1) + 1):
            num[k] += np.sum(s[:-k] * s[k:])
    rho = num / den
    n_corr = 1.
    for k in range(1, kmax + 1):
        if rho[k] <= 0:
            break
        n_corr += 2 * rho[k]
    return n_corr, rho[1:]

def measure_error_scale(x, y, err, ranges=None, deg=1, relative=False, step=0.01):
    """Rescale err from the scatter about a baseline in `ranges` (see baseline_residuals). Returns (err * factor, factor)."""
    f = error_scale_from_residuals(baseline_residuals(x, y, err, ranges, deg, relative), step)
    return f * np.asarray(err, dtype=float), f

def measure_n_corr(x, y, err, ranges=None, deg=1, relative=False, kmax=15):
    """n_corr from the residuals about a baseline in `ranges` (see baseline_residuals). Returns (n_corr, rho)."""
    return n_corr_from_residuals(baseline_residuals(x, y, err, ranges, deg, relative), kmax)
