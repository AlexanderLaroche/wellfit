import numpy as np

# Group-by-group diagnostics and systematic errors from the per-group chi2_eff matrix C (n_models, n_groups)
# (wellfit.stats.group_chi2_eff): leave-one-out fits, tension, and a stratified bootstrap over groups.
# A group is any set of data points that one would trust or distrust together (a spectral line, a photometric band,
# an observing epoch, an instrument, a detector chip, ...).

def leave_one_out(C):
    """Best model index when each group is left out: argmin over models of sum_{i != j} C[:, i]. Returns (n_groups,)."""
    C = np.asarray(C)
    return np.argmin(C.sum(axis=1)[:, None] - C, axis=0)

def tension(C, best=None):
    """
    Tension of each group: how much the fit to all the other groups improves (drop in their chi2_eff) when this group
    is left out and the models refitted. ~0 for a group consistent with the others; large values flag a group that
    pulls the fit away from where the rest of the data want it (a model deficiency, a calibration problem, ...).
    best: index of the best model with all groups (default: argmin of the total).
    """
    C = np.asarray(C)
    tot = C.sum(axis=1)
    best = np.argmin(tot) if best is None else best
    others = tot[:, None] - C
    return others[best] - others.min(axis=0)

def bootstrap_weights(strata, n_boot, seed=0):
    """
    Stratified bootstrap over groups: in each replicate the groups of each stratum are resampled with replacement,
    keeping the stratum sizes. strata: one label per group (all equal = ordinary bootstrap over groups).
    Returns group weights (n_groups, n_boot).
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    W = np.zeros((strata.size, n_boot), dtype=np.float32)
    for f in np.unique(strata):
        idx = np.flatnonzero(strata == f)
        draws = rng.integers(0, idx.size, size=(idx.size, n_boot))
        for i in range(n_boot):
            W[:, i] += np.bincount(idx[draws[:, i]], minlength=strata.size)
    return W

def stratum_drop_weights(strata):
    """Group weights (n_groups, n_strata) leaving out one stratum at a time, and the stratum names (order of appearance)."""
    strata = np.asarray(strata)
    names = [str(f) for f in dict.fromkeys(strata)]
    return np.stack([(strata != f).astype(np.float32) for f in names], axis=1), names

def refit_weighted(C, W):
    """Best model index for each column of group weights W (n_groups, n_rep): argmin over models of C @ W."""
    return np.argmin(np.asarray(C) @ W, axis=0)
