"""
Toy line-broadening model and synthetic data for the wellfit example notebooks.

Four generic absorption lines A-D, each a Gaussian of known equivalent width, centre and intrinsic width, broadened by
rotation (v sin i, Gray profile, linear limb darkening eps = 0.6) and a Gaussian "macroturbulence" (v_macro). The kernels are
only there to demonstrate the method; swap in your own model.
"""
import numpy as np
from scipy.special import j1

C_KMS = 2.99792458e5
DV = 1.0                                   # km/s, velocity grid of the convolutions
V = np.arange(-1500., 1500., DV)
K = 2 * np.pi * np.fft.rfftfreq(V.size, DV)  # angular frequency [rad / (km/s)]

LINES = {  # name: (centre [A], equivalent width [A])
    'A': (4000., 0.30),
    'B': (4300., 0.20),
    'C': (4600., 0.25),
    'D': (4900., 0.30),
}
SIGMA0 = 10.     # km/s, intrinsic (known) Gaussian width of every line
HALF_WIDTH = 15. # A, half width of the window cut around each line
CONT_EDGE = 6.   # A, pixels beyond this distance from the centre are line-free continuum

# The kernels are applied in Fourier space through their analytic transforms, so that the model is a smooth function
# of v sin i and v_macro (a kernel sampled on the velocity grid changes in steps, which breaks finite-difference
# derivatives in the optimizer).

def rotation_ft(vsini, eps=0.6):
    """Fourier transform of the rotation profile (Gray), normalized to 1 at zero frequency."""
    z = K * abs(vsini)
    small = z < 1e-2                                 # series expansions near zero frequency (cancellation)
    zs = np.where(small, 1., z)
    semi = np.where(small, 1 - z**2 / 8, 2 * j1(zs) / zs)                                  # transform of sqrt(1 - x^2)
    para = np.where(small, 1 - z**2 / 10, 3 * (np.sin(zs) - zs * np.cos(zs)) / zs**3)      # transform of 1 - x^2
    w1, w2 = np.pi * (1 - eps), 2 * np.pi * eps / 3  # areas of the two terms
    return (w1 * semi + w2 * para) / (w1 + w2)

def gaussian_ft(sigma):
    """Fourier transform of a unit-area Gaussian of width sigma."""
    return np.exp(-0.5 * (K * sigma)**2)

def line_profile(lam, centre, ew, vsini, vmacro):
    """Normalized flux of one line at wavelengths lam."""
    # intrinsic absorption in velocity, normalized so that its integral over wavelength is ew
    a = ew * C_KMS / centre * np.exp(-0.5 * (V / SIGMA0)**2) / (np.sqrt(2 * np.pi) * SIGMA0)
    a = np.fft.irfft(np.fft.rfft(np.fft.ifftshift(a)) * rotation_ft(vsini) * gaussian_ft(vmacro), n=V.size)
    return 1. - np.interp(C_KMS * (lam / centre - 1.), V, np.fft.fftshift(a), left=0., right=0.)

def model(lam, groups, vsini, vmacro, names=tuple(LINES)):
    """All lines: lam and groups (line index of every pixel) as returned by make_data."""
    out = np.empty_like(lam)
    for j, n in enumerate(names):
        m = groups == j
        out[m] = line_profile(lam[m], *LINES[n], vsini, vmacro)
    return out

def correlated_noise(n, s, rng):
    """Unit-variance Gaussian noise correlated over ~s pixels (white noise smoothed with a Gaussian of s pixels)."""
    k = np.exp(-0.5 * (np.arange(-5 * s, 5 * s + 1) / s)**2)
    return np.convolve(rng.normal(size=n + k.size), k / np.sqrt(np.sum(k**2)), mode='valid')[:n]

def make_data(vsini=75., vmacro=20., vsini_D=150., snr=80., smooth_px=1.5, err_misquote=0.8, dlam=0.1, seed=42):
    """
    Synthetic spectrum: windows around lines A-D sampled every dlam A. Noise of 1/snr per pixel, correlated over
    ~smooth_px pixels (as after resampling or stacking). The quoted errors are err_misquote times the true noise (as
    after an imperfect reduction). Line D is generated with vsini_D instead of vsini, standing in for any problem a
    single line can have (a blend, a poor model of that line, a normalization error).
    """
    rng = np.random.default_rng(seed)
    lam, groups, ref = [], [], []
    for j, (c, _) in enumerate(LINES.values()):
        x = np.arange(c - HALF_WIDTH, c + HALF_WIDTH + dlam / 2, dlam)
        lam.append(x); groups.append(np.full(x.size, j))
        ref += [(c - HALF_WIDTH - 1, c - CONT_EDGE), (c + CONT_EDGE, c + HALF_WIDTH + 1)]
    lam, groups = np.concatenate(lam), np.concatenate(groups)
    true = model(lam, groups, vsini, vmacro)
    D = groups == 3
    true[D] = line_profile(lam[D], *LINES['D'], vsini_D, vmacro)
    noise = np.concatenate([correlated_noise(np.sum(groups == j), smooth_px, rng) for j in range(len(LINES))]) / snr
    flux = true + noise
    err = np.full(lam.size, err_misquote / snr)
    truth = {'vsini': vsini, 'vmacro': vmacro, 'vsini_D': vsini_D,
             'n_corr': 1 + 2 * np.sum(np.exp(-np.arange(1, 16)**2 / (4 * smooth_px**2)))}
    return dict(lam=lam, flux=flux, err=err, groups=groups, ref_ranges=ref, names=list(LINES), truth=truth)
