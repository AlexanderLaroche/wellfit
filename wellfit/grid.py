import numpy as np

# Parameter errors on a model grid, respecting the grid resolution (no error smaller than one grid step).

def grid_values(vals):
    return np.unique(np.round(np.asarray(vals, dtype=float), 6))

def snap(x, vals):
    """Nearest grid value."""
    g = grid_values(vals)
    return g[np.argmin(np.abs(g - x))]

def grid_steps(vals, v):
    """Distances from v to the neighboring values of a grid axis, (below, above); 0 at the grid edges."""
    g = grid_values(vals)
    i = np.argmin(np.abs(g - v))
    return (v - g[i-1] if i > 0 else 0.), (g[i+1] - v if i < g.size - 1 else 0.)

def weighted_quantiles(x, w, q=(0.16, 0.84)):
    """
    Quantiles of a weighted distribution of grid values x (e.g. the marginal posterior of one parameter over a
    flattened multi-dimensional grid). Each distinct value is spread uniformly over a cell reaching halfway to its
    neighbors (cells end at the outermost values).
    """
    xu, inv = np.unique(np.round(np.asarray(x, dtype=float), 8), return_inverse=True)
    wu = np.bincount(inv, weights=w)
    if xu.size == 1:
        return np.full(len(q), xu[0])
    edges = np.concatenate([[xu[0]], 0.5 * (xu[1:] + xu[:-1]), [xu[-1]]])
    cdf = np.concatenate([[0.], np.cumsum(wu)]) / wu.sum()
    return np.interp(q, cdf, edges)

def stat_error(vals, post, best):
    """
    Statistical error of a grid parameter: 16-84th percentiles of its marginal posterior, snapped to the grid, at least
    one grid step on each side (where the grid continues). vals: the parameter of every model; post: posterior weights.
    Returns (value, +err, -err).
    """
    vals = np.asarray(vals, dtype=float)
    c = vals[best]
    q16, q84 = (snap(q, vals) for q in weighted_quantiles(vals, post))
    st_lo, st_hi = grid_steps(vals, c)
    return c, max(q84 - c, st_hi), max(c - q16, st_lo)

def sys_error(rep_vals, best_val, vals):
    """
    Systematic error from refitted best values (e.g. over line-bootstrap replicates): 16-84th percentiles relative to
    the best value, snapped to the grid. Returns (+sys, -sys).
    """
    q16, q84 = np.percentile(np.asarray(rep_vals, dtype=float), [16, 84])
    up, dn = max(snap(q84, vals) - best_val, 0.), max(best_val - snap(q16, vals), 0.)
    return (up if up > 1e-6 else 0.), (dn if dn > 1e-6 else 0.)
