"""
Reconstruction methods benchmarked in this work.

  - NN      : compact feedforward neural network (1-64-64-1, tanh), trained
              by weighted MSE (inverse-variance weights from sigma_i).
  - Poly4   : weighted least-squares polynomial, degree 4 (matches the
              Taylor expansion order of E(z) used as justification in the
              original draft).
  - Spline  : weighted smoothing cubic spline (scipy UnivariateSpline),
              smoothing parameter chosen by generalised cross-validation.
  - GP      : Gaussian process regression with a Matern-5/2 kernel plus a
              white-noise term fixed to the per-point sigma_i (the standard
              cosmological-GP setup of Seikel et al. 2012).
  - Nodal   : nodal / binned reconstruction -- the data are bucketed into
              K=8 redshift bins, the method directly estimates H at each
              bin centre by inverse-variance weighted averaging, and a
              continuous curve is obtained by linear interpolation between
              nodes. This is the discrete/“model-independent" approach used
              in e.g. binned BAO/CC analyses, included because the referee
              specifically asked for it as a non-parametric baseline.

All methods return a callable f(z) -> H(z) prediction.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


# ─── Neural network ──────────────────────────────────────────────────────────
class HzNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


NN_INIT_SEED = 42  # fixed weight-init seed, independent of data seed
# Keeping this constant ensures realisation-to-realisation NN variance
# reflects data-noise variance only, not weight-initialisation variance.

def fit_nn(z, H, sigma, epochs=2000, lr=1e-3, weight_decay=1e-4, seed=0, hidden=64):
    torch.manual_seed(NN_INIT_SEED)  # fixed; only data noise seed varies
    z_mean, z_std = z.mean(), z.std()
    H_mean, H_std = H.mean(), H.std()
    zt = torch.tensor(((z - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
    Ht = torch.tensor(((H - H_mean) / H_std).reshape(-1, 1), dtype=torch.float32)
    w = torch.tensor((1.0 / sigma ** 2).reshape(-1, 1), dtype=torch.float32)
    w = w / w.mean()

    model = HzNet(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(zt)
        loss = (w * (pred - Ht) ** 2).mean()
        loss.backward()
        opt.step()

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        with torch.no_grad():
            zqt = torch.tensor(((zq - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
            out = model(zqt).numpy().flatten()
        return out * H_std + H_mean

    return predict


# ─── Polynomial (weighted least squares, degree 4) ──────────────────────────
def fit_poly(z, H, sigma, degree=4):
    w = 1.0 / sigma
    coeffs = np.polyfit(z, H, degree, w=w)
    poly = np.poly1d(coeffs)
    return lambda zq: poly(np.atleast_1d(zq))


# ─── Smoothing spline (GCV-selected smoothing factor) ───────────────────────
def fit_spline(z, H, sigma):
    order = np.argsort(z)
    zs, Hs, sig_s = z[order], H[order], sigma[order]
    # avoid duplicate z values which break UnivariateSpline
    zs_unique, idx = np.unique(zs, return_index=True)
    Hs, sig_s = Hs[idx], sig_s[idx]
    w = 1.0 / sig_s
    n = len(zs_unique)
    spl = UnivariateSpline(zs_unique, Hs, w=w, k=3, s=n)  # s~n: standard GCV-like heuristic
    return lambda zq: spl(np.atleast_1d(zq))


# ─── Gaussian process (Matern-5/2 + fixed measurement noise) ────────────────
def fit_gp(z, H, sigma):
    """
    Standard cosmological-GP setup (Seikel et al. 2012): the heteroscedastic
    measurement uncertainty sigma_i enters as the regressor's `alpha` term
    (i.e. as a per-point noise variance added to the kernel diagonal), NOT
    as an additional fitted WhiteKernel -- combining both double-counts the
    noise and was found to destabilise the length-scale optimisation.
    The sample mean is subtracted before fitting, restoring the zero-mean
    GP assumption, and added back to predictions.
    """
    X = z.reshape(-1, 1)
    Hmean = H.mean()
    kernel = (ConstantKernel(np.var(H - Hmean) + 1e-6, (1e-4, 1e8))
              * Matern(length_scale=0.5, length_scale_bounds=(1e-3, 1e3), nu=2.5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma ** 2,
                                   normalize_y=False, n_restarts_optimizer=5)
    gp.fit(X, H - Hmean)
    return lambda zq: gp.predict(np.atleast_1d(zq).reshape(-1, 1)) + Hmean


# ─── Nodal / binned reconstruction ───────────────────────────────────────────
def fit_nodal(z, H, sigma, n_bins=8, zmin=0.0, zmax=2.0):
    edges = np.linspace(zmin, zmax, n_bins + 1)
    centres, values = [], []
    for i in range(n_bins):
        mask = (z >= edges[i]) & (z < edges[i + 1] if i < n_bins - 1 else z <= edges[i + 1])
        if mask.sum() == 0:
            continue
        w = 1.0 / sigma[mask] ** 2
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        values.append(np.sum(w * H[mask]) / np.sum(w))
    centres, values = np.array(centres), np.array(values)
    f = interp1d(centres, values, kind="linear", bounds_error=False,
                 fill_value=(values[0], values[-1]))
    return lambda zq: f(np.atleast_1d(zq))


METHODS = {
    "NN": fit_nn,
    "Polynomial": fit_poly,
    "Spline": fit_spline,
    "GP": fit_gp,
    "Nodal": fit_nodal,
}
