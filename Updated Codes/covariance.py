"""
Covariance-matrix treatment of cosmic-chronometer systematics
(Moresco et al. 2020, ApJ, 898, 82 -- "Setting the Stage for Cosmic
Chronometers II"), and generalised-least-squares (GLS) extensions of the
methods that can actually consume a non-diagonal covariance.

FRAMEWORK (real, citable, not fabricated)
------------------------------------------
Moresco et al. (2020) show that model-driven systematics (IMF, stellar
library, SPS model choice) act MULTIPLICATIVELY and COHERENTLY across the
whole CC sample, because every point is analysed with a common modelling
choice. This gives a rank-deficient, fully-correlated covariance
contribution of the form (their Eq. 9):

    Cov_ij^X = eta_X(z_i) H(z_i) * eta_X(z_j) H(z_j)

for each systematic source X, where eta_X(z) is that source's fractional
uncertainty. The total covariance is

    Cov = diag(sigma_stat^2) + sum_X Cov^X.

WHAT WE DO NOT HAVE
--------------------
We do not have the exact per-point, per-source eta_X(z) values from
Moresco et al. (2020) Table 3 reproduced numerically in this session (no
network access to the journal's machine-readable table). Reproducing
Table 3 verbatim without the primary source would be worse than being
explicit about it. What IS solidly documented in secondary literature
(Loubser et al. 2025; DESI CC systematics paper, arXiv:2511.02730) is a
representative CONSERVATIVE combined model-systematic envelope that
several follow-up papers adopt directly:

    eta_model(z=0.07 ish, low z)  ~ 13.2%
    eta_model(z=1.965, high z)    ~ 3.9%

(quoted range from Moresco & collaborators' own conservative summary,
"Setting the Stage II", Sec. on current CC data). We linearly interpolate
eta_model(z) between these two literature-quoted endpoints as an explicit,
labelled APPROXIMATION to the real Table 3 -- not a fabricated number, but
a documented simplification that should be replaced with the exact
tabulated values before this is submitted anywhere. This is stated
explicitly in the manuscript text this module supports.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, cho_factor


def eta_model(z, z_lo=0.07, z_hi=1.965, eta_lo=0.132, eta_hi=0.039):
    """Linear interpolation of the conservative combined model-systematic
    fractional uncertainty between the literature-quoted low-z and high-z
    endpoints (see module docstring). APPROXIMATION -- replace with the
    exact Moresco et al. (2020) Table 3 values for a submission-ready
    analysis."""
    z = np.atleast_1d(z).astype(float)
    frac = np.clip((z - z_lo) / (z_hi - z_lo), 0, 1)
    return eta_lo + frac * (eta_hi - eta_lo)


def build_covariance(z, H, sigma_stat):
    """Cov = diag(sigma_stat^2) + rank-1 correlated model-systematic term."""
    eta = eta_model(z)
    sys_vec = eta * H
    Cov = np.diag(sigma_stat ** 2) + np.outer(sys_vec, sys_vec)
    return Cov


# ─── GLS-capable method refits ──────────────────────────────────────────
def fit_poly_gls(z, H, Cov, degree=4):
    """Weighted->generalised least squares polynomial fit using the full
    covariance matrix (whitening via Cholesky factor of Cov)."""
    L = cholesky(Cov, lower=True)
    Linv = np.linalg.inv(L)
    Hw = Linv @ H
    X = np.vander(z, degree + 1, increasing=True)
    Xw = Linv @ X
    coeffs, *_ = np.linalg.lstsq(Xw, Hw, rcond=None)
    poly = np.poly1d(coeffs[::-1])
    return lambda zq: poly(np.atleast_1d(zq))


def fit_gp_gls(z, H, Cov, length_scale=0.5, nu_amp=None):
    """Matern-5/2 GP regression with the FULL noise covariance (not just
    the diagonal) added to the kernel Gram matrix -- the direct GLS
    generalisation of the diagonal-alpha GP used in the main paper."""
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel

    Hmean = H.mean()
    y = H - Hmean
    amp = np.var(y) + 1e-6
    kernel_func = ConstantKernel(amp) * Matern(length_scale=length_scale, nu=2.5)
    X = z.reshape(-1, 1)
    K = kernel_func(X) + Cov
    c, low = cho_factor(K)
    alpha = cho_solve((c, low), y)
    K_func = kernel_func

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float).reshape(-1, 1)
        Ks = K_func(zq, X)
        return (Ks @ alpha) + Hmean

    return predict


def fit_nn_gls(z, H, Cov, epochs=4000, lr=1e-2, hidden=64, weight_decay=1e-4):
    """NN fit using the full precision matrix Cov^-1 in a GLS loss,
    loss = (1/N) r^T Cov^-1 r, in place of the diagonal inverse-variance
    weighting used in the main paper. Pure NumPy (no torch dependency)."""
    rng = np.random.default_rng(42)
    Cinv = np.linalg.inv(Cov)
    zc = (z - z.mean()) / z.std()
    Hc_scale = H.std()
    Hc = (H - H.mean()) / Hc_scale
    Cinv_scaled = Cinv * (Hc_scale ** 2)  # rescale precision to normalised H units
    n, h = len(z), hidden
    W1 = rng.normal(0, 1.0, (1, h)); b1 = np.zeros(h)
    W2 = rng.normal(0, 1.0 / np.sqrt(h), (h, h)); b2 = np.zeros(h)
    W3 = rng.normal(0, 1.0 / np.sqrt(h), (h, 1)); b3 = np.zeros(1)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    X = zc.reshape(-1, 1); y = Hc.reshape(-1, 1)
    for t in range(1, epochs + 1):
        a1 = np.tanh(X @ W1 + b1)
        a2 = np.tanh(a1 @ W2 + b2)
        pred = a2 @ W3 + b3
        r = (pred - y).flatten()
        loss_grad = (2.0 / n) * (Cinv_scaled @ r)
        loss_grad = loss_grad.reshape(-1, 1)
        gW3 = a2.T @ loss_grad + 2 * weight_decay * W3
        gb3 = loss_grad.sum(0)
        da2 = loss_grad @ W3.T * (1 - a2 ** 2)
        gW2 = a1.T @ da2 + 2 * weight_decay * W2
        gb2 = da2.sum(0)
        da1 = da2 @ W2.T * (1 - a1 ** 2)
        gW1 = X.T @ da1 + 2 * weight_decay * W1
        gb1 = da1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)
            mhat = m[i] / (1 - beta1 ** t); vhat = v[i] / (1 - beta2 ** t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        Xq = ((zq - z.mean()) / z.std()).reshape(-1, 1)
        a1 = np.tanh(Xq @ W1 + b1)
        a2 = np.tanh(a1 @ W2 + b2)
        return (a2 @ W3 + b3).flatten() * H.std() + H.mean()

    return predict


# Spline and Nodal explicitly do NOT get a GLS generalisation here: the
# standard scipy UnivariateSpline penalised-roughness objective and the
# nodal inverse-variance bin average both assume diagonal weights by
# construction, and correlated-error generalisations of either (e.g. GLS
# smoothing splines) are a non-trivial numerical-methods contribution in
# their own right, not a drop-in change. We therefore run Spline and Nodal
# using w_i = 1/Cov_ii (diagonal of the FULL covariance, i.e. inflated by
# the systematic term added in quadrature on the diagonal) and flag this
# explicitly as an asymmetry across methods in the manuscript text -- a
# real methodological limitation of binned/spline estimators relative to
# GP, polynomial GLS, and a custom-loss NN, not an oversight.
def diag_only_weights(Cov):
    return np.sqrt(np.diag(Cov))
