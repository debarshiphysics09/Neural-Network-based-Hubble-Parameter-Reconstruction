"""
Synthetic observation generator.

Three observable types are supported, matching the three main background
probes used in real H(z)-reconstruction work:

  - "Hz"  : direct H(z) measurements (cosmic chronometers / radial BAO),
            redshift sampling matches the real compilation of Moresco et al.
            (2022), which has ~32 cosmic-chronometer points over 0<z<2;
            we additionally test N=80 to anticipate near-future compilations
            combining CC + radial BAO (DESI, MOONS).
  - "DH"  : the Hubble distance D_H(z) = c/H(z), the line-of-sight BAO
            observable, with noise added as a fixed *relative* fractional
            uncertainty (typical of BAO measurements, ~2-5%).
  - "mu"  : the supernova distance modulus mu(z) = 5 log10(D_L/10pc),
            with noise at the level of current Pantheon+-like compilations
            (~0.1-0.15 mag per SN after binning).

Redshift sampling: a Beta-distributed grid over [0.01, 2] concentrating
points at low-to-intermediate z, matching the non-uniform redshift
distribution of real CC/BAO catalogues (survey-depth limited).
"""
import numpy as np
from scipy.signal import savgol_filter
from cosmology import C_LIGHT


def redshift_grid(n, zmin=0.01, zmax=2.0, seed=None, clustered=True):
    rng = np.random.default_rng(seed)
    if not clustered:
        return np.sort(rng.uniform(zmin, zmax, n))
    u = rng.beta(1.6, 2.2, n)
    z = zmin + u * (zmax - zmin)
    return np.sort(z)


def generate_Hz_data(cosmology, n=80, sigma=3.0, zmin=0.01, zmax=2.0, seed=0):
    """Direct H(z) measurements with absolute Gaussian noise sigma (km/s/Mpc)."""
    rng = np.random.default_rng(seed)
    z = redshift_grid(n, zmin, zmax, seed=seed)
    H_true = cosmology.Hz(z)
    H_obs = H_true + rng.normal(0, sigma, n)
    sig = np.full(n, sigma)
    return z, H_obs, sig, H_true


def generate_DH_data(cosmology, n=80, rel_sigma=0.03, zmin=0.01, zmax=2.0, seed=0):
    """BAO Hubble-distance proxy D_H(z)=c/H(z), fractional noise rel_sigma,
    propagated back to an effective H(z) measurement + 1-sigma error."""
    rng = np.random.default_rng(seed)
    z = redshift_grid(n, zmin, zmax, seed=seed)
    DH_true = cosmology.D_H(z)
    DH_obs = DH_true * (1 + rng.normal(0, rel_sigma, n))
    H_obs = C_LIGHT / DH_obs
    sigma_DH = rel_sigma * DH_true
    sigma_H = (C_LIGHT / DH_true ** 2) * sigma_DH
    H_true = cosmology.Hz(z)
    return z, H_obs, sigma_H, H_true


def generate_mu_data(cosmology, n=80, sigma_mu=0.12, zmin=0.01, zmax=2.0, seed=0):
    """SN-like distance modulus mu(z), converted to an effective H(z)
    measurement via smoothed numerical differentiation of D_C(z)."""
    rng = np.random.default_rng(seed)
    z = redshift_grid(n, zmin, zmax, seed=seed)
    mu_true = cosmology.distance_modulus(z)
    mu_obs = mu_true + rng.normal(0, sigma_mu, n)
    DL_obs = 10 ** ((mu_obs - 25) / 5)
    DC_obs = DL_obs / (1 + z)
    order = np.argsort(z)
    zs, DCs = z[order], DC_obs[order]
    win = max(5, (n // 4) | 1)
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)
    DCs_smooth = savgol_filter(DCs, window_length=win, polyorder=2)
    dDCdz = np.gradient(DCs_smooth, zs)
    H_eff = C_LIGHT / np.clip(dDCdz, 1e-3, None)
    H_smooth_trend = savgol_filter(H_eff, win, 2)
    sigma_H = np.full(n, np.std(H_eff - H_smooth_trend) + 1.0)
    H_true_out = cosmology.Hz(zs)
    return zs, H_eff, sigma_H, H_true_out


GENERATORS = {
    "Hz": generate_Hz_data,
    "DH": generate_DH_data,
    "mu": generate_mu_data,
}
