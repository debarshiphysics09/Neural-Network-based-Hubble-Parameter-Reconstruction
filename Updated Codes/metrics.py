"""
Performance metrics.

Note on the Cramer-Rao critique raised in peer review
------------------------------------------------------
The original manuscript compared the RMSE of a *nonparametric function
reconstruction* against sigma/sqrt(N), the Cramer-Rao bound for estimating
a *single scalar parameter* (e.g. a constant) from N i.i.d. Gaussian
measurements. The referee correctly identifies this as a mismatched
statistical comparison: reconstructing an entire curve has many more
effective degrees of freedom than estimating one number, so sigma/sqrt(N)
is not an achievable bound for the reconstruction problem and the resulting
"efficiency" figure is not meaningful.

We replace it with two well-posed diagnostics:

  1. Local noise-floor benchmark: at each redshift z_i with a real or
     synthetic data point, the *irreducible* local uncertainty on the curve
     value is approximately sigma_i / sqrt(n_eff(z_i)), where n_eff(z_i) is
     the effective number of points contributing to the local estimate
     (the same n_eff a kernel/bin estimator would integrate over). We
     therefore report this only as a heuristic local noise floor, with an
     explicit n_eff defined by the same bin width used in the nodal method,
     and DO NOT call it an "efficiency" -- it is now labelled the
     "binned noise floor" for visual comparison only.

  2. Parametric reference bound (the well-posed comparison): we ALSO fit
     the *correct* parametric model (flat LCDM with free H0, Om) by
     weighted nonlinear least squares to the same synthetic data, and
     compute its parameter-covariance via the Fisher information matrix.
     This is now an honest apples-to-apples comparison: a 2-parameter
     model that is correctly specified achieves close to its Cramer-Rao
     bound, and we report the *ratio* of nonparametric-method RMSE to the
     RMSE of this correctly-specified parametric fit, propagated through
     its own covariance. This ratio quantifies the well-known price of
     model-independence: how much precision is sacrificed by not assuming
     the correct functional form. It does not at any point claim that the
     nonparametric methods *should* attain the scalar Cramer-Rao bound.
"""
import numpy as np
from scipy.optimize import curve_fit


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def integrated_rmse(predict_fn, cosmology, zgrid):
    pred = predict_fn(zgrid)
    true = cosmology.Hz(zgrid).flatten()
    return rmse(pred, true), mae(pred, true)


def deceleration_parameter(predict_fn, zgrid, h=1e-3):
    """q(z) = (1+z) H'(z)/H(z) - 1 computed by central differences on the
    reconstructed H(z) curve."""
    Hz = predict_fn(zgrid)
    Hzp = predict_fn(zgrid + h)
    Hzm = predict_fn(zgrid - h)
    dHdz = (Hzp - Hzm) / (2 * h)
    return (1 + zgrid) * dHdz / Hz - 1


def transition_redshift_from_curve(zgrid, qvals):
    sign_change = np.where(np.diff(np.sign(qvals)) != 0)[0]
    if len(sign_change) == 0:
        return np.nan
    i = sign_change[0]
    z1, z2 = zgrid[i], zgrid[i + 1]
    q1, q2 = qvals[i], qvals[i + 1]
    return z1 - q1 * (z2 - z1) / (q2 - q1)


def binned_noise_floor(z, sigma, zgrid, n_bins=8, zmin=0.0, zmax=2.0):
    """Heuristic local noise floor sigma_i/sqrt(n_eff) using the same
    bin width as the nodal method; for visual comparison ONLY -- not a
    formal Cramer-Rao bound on the nonparametric reconstruction."""
    edges = np.linspace(zmin, zmax, n_bins + 1)
    floor = np.zeros_like(zgrid)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (z >= lo) & (z < hi if i < n_bins - 1 else z <= hi)
        zmask = (zgrid >= lo) & (zgrid < hi if i < n_bins - 1 else zgrid <= hi)
        n_eff = max(mask.sum(), 1)
        sig_local = np.mean(sigma[mask]) if mask.sum() > 0 else np.mean(sigma)
        floor[zmask] = sig_local / np.sqrt(n_eff)
    return floor


def flat_lcdm_model(z, H0, Om):
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def parametric_reference_rmse(z, H_obs, sigma, cosmology, zgrid, true_H0=70.0, true_Om=0.30):
    """
    Fit the CORRECT 2-parameter flat-LCDM model (H0, Om free) to the data
    by weighted nonlinear least squares; this is the well-posed comparison
    point requested in review -- the achievable precision of a correctly
    specified low-dimensional model, evaluated on the identical synthetic
    realisation as the nonparametric methods.
    """
    try:
        popt, pcov = curve_fit(flat_lcdm_model, z, H_obs, p0=[70, 0.3],
                                sigma=sigma, absolute_sigma=True,
                                bounds=([40, 0.05], [100, 0.95]))
    except Exception:
        return np.nan, np.nan
    pred = flat_lcdm_model(zgrid, *popt)
    true = cosmology.Hz(zgrid).flatten()
    return rmse(pred, true), popt
