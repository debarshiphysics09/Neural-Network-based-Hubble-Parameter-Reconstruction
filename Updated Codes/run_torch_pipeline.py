"""
PyTorch real-data pipeline -- run this locally where torch is installed.

WHY THIS FILE EXISTS
---------------------
Every real-data result reported so far (Sec. 5 of manuscript_revised.tex)
used a NumPy reimplementation of the neural network (`fit_nn_numpy` in
run_real_data.py), because torch was not installed in the sandboxed
environment. This was explicitly flagged as an unverified limitation.
This script closes that gap: it uses the REAL `fit_nn` from methods.py
(actual PyTorch, actual Adam, identical architecture/loss/seed) for
every NN result, and additionally runs a consistency check against the
NumPy version so you can confirm (or refute) that the two implementations
actually agree before trusting any downstream conclusion built on them.

WHAT TO DO
-----------
1. pip install torch scikit-learn scipy pandas matplotlib
2. python3 run_torch_pipeline.py
3. Send me back the console output and the four CSV files it writes:
     - torch_vs_numpy_consistency.csv
     - real_data_loocv_torch.csv
     - real_data_bao_combined_loocv_torch.csv
     - real_data_covariance_loocv_torch.csv
   (plus the two regenerated PNGs, if you want the figures updated too)

WHAT THIS SCRIPT DOES, STEP BY STEP
-------------------------------------
A. Consistency check: fits both fit_nn (torch) and fit_nn_numpy on the
   same synthetic dataset and reports the RMSE between the two curves'
   predictions on a common grid, plus their individual RMSE against the
   known synthetic ground truth. If these two numbers are close, the
   NumPy-based results already reported are validated retroactively.
B. Re-runs the real-CC-data LOOCV (Sec. 5.2 of the manuscript) with the
   real torch NN, including bootstrap 68% CIs and the paired
   significance test against the next-best method.
C. Re-runs the real-BAO / combined LOOCV (Sec. 5.3) the same way.
D. Re-runs the GLS covariance-aware NN fit (Sec. 5.4) using a
   torch-based precision-weighted loss in place of the NumPy version.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from methods import fit_nn, fit_poly, fit_spline, fit_gp, fit_nodal
from cosmology import FLAT_LCDM
from data_gen import generate_Hz_data
from metrics import integrated_rmse, deceleration_parameter, transition_redshift_from_curve, flat_lcdm_model
from real_data import load_cc_data
from real_bao_data import load_bao_data
from bootstrap_stats import bootstrap_chi_ci, paired_bootstrap_pvalue
from scipy.optimize import curve_fit

ZGRID_SYN = np.linspace(0, 2, 400)


# ── A. Torch vs NumPy consistency check ─────────────────────────────────
def fit_nn_numpy(z, H, sigma, epochs=2000, lr=1e-3, hidden=64, seed=0, weight_decay=1e-4):
    """Copy of the NumPy fallback used in run_real_data.py, kept here only
    for the consistency check -- NOT used for any reported result in
    this script."""
    rng = np.random.default_rng(42)
    z_mean, z_std = z.mean(), z.std()
    H_mean, H_std = H.mean(), H.std()
    zc = (z - z_mean) / z_std
    Hc = (H - H_mean) / H_std
    w = (1.0 / sigma ** 2); w = w / w.mean()
    n, h = len(z), hidden
    W1 = rng.normal(0, 1.0, (1, h)); b1 = np.zeros(h)
    W2 = rng.normal(0, 1.0 / np.sqrt(h), (h, h)); b2 = np.zeros(h)
    W3 = rng.normal(0, 1.0 / np.sqrt(h), (h, 1)); b3 = np.zeros(1)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    X = zc.reshape(-1, 1); y = Hc.reshape(-1, 1); ww = w.reshape(-1, 1)
    for t in range(1, epochs + 1):
        a1 = np.tanh(X @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2); pred = a2 @ W3 + b3
        err = pred - y
        loss_grad = 2 * ww * err / n
        gW3 = a2.T @ loss_grad + 2 * weight_decay * W3; gb3 = loss_grad.sum(0)
        da2 = loss_grad @ W3.T * (1 - a2 ** 2)
        gW2 = a1.T @ da2 + 2 * weight_decay * W2; gb2 = da2.sum(0)
        da1 = da2 @ W2.T * (1 - a1 ** 2)
        gW1 = X.T @ da1 + 2 * weight_decay * W1; gb1 = da1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g ** 2)
            mhat = m[i] / (1 - beta1 ** t); vhat = v[i] / (1 - beta2 ** t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        Xq = ((zq - z_mean) / z_std).reshape(-1, 1)
        a1 = np.tanh(Xq @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2)
        return (a2 @ W3 + b3).flatten() * H_std + H_mean

    return predict


def run_consistency_check():
    print("=" * 70)
    print("A. Torch vs NumPy NN consistency check (synthetic data, sigma=3)")
    print("=" * 70)
    z, Hobs, sig, Htrue = generate_Hz_data(FLAT_LCDM, n=80, sigma=3.0, seed=0)
    pred_torch = fit_nn(z, Hobs, sig, seed=0)
    pred_numpy = fit_nn_numpy(z, Hobs, sig, seed=0)
    rmse_torch, _ = integrated_rmse(pred_torch, FLAT_LCDM, ZGRID_SYN)
    rmse_numpy, _ = integrated_rmse(pred_numpy, FLAT_LCDM, ZGRID_SYN)
    cross_rmse = float(np.sqrt(np.mean((pred_torch(ZGRID_SYN) - pred_numpy(ZGRID_SYN)) ** 2)))
    print(f"  Torch NN RMSE vs ground truth : {rmse_torch:.4f} km/s/Mpc")
    print(f"  NumPy NN RMSE vs ground truth : {rmse_numpy:.4f} km/s/Mpc")
    print(f"  Torch-vs-NumPy prediction RMSE (curve-to-curve) : {cross_rmse:.4f} km/s/Mpc")
    verdict = ("CONSISTENT (curve-to-curve RMSE is small relative to both methods' "
               "ground-truth RMSE)" if cross_rmse < 0.5 * max(rmse_torch, rmse_numpy)
               else "NOT CLOSELY CONSISTENT -- treat prior NumPy-based real-data "
                    "results as provisional and prefer this torch run")
    print(f"  Verdict: {verdict}")
    pd.DataFrame([dict(rmse_torch=rmse_torch, rmse_numpy=rmse_numpy,
                        cross_rmse=cross_rmse, verdict=verdict)]
                 ).to_csv("torch_vs_numpy_consistency.csv", index=False)
    return pred_torch


# ── B/C. Real-data LOOCV with the real torch NN ─────────────────────────
METHODS = {"NN": fit_nn, "Polynomial": fit_poly, "Spline": fit_spline,
           "GP": fit_gp, "Nodal": fit_nodal}


def loocv(fitter, name, z, H, sigma):
    n = len(z)
    resid = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            if name == "NN":
                predict = fitter(z[mask], H[mask], sigma[mask], seed=0)
            else:
                predict = fitter(z[mask], H[mask], sigma[mask])
            pred_i = predict(z[i])[0]
        except Exception as e:
            print(f"    [{name}] LOOCV point {i} failed: {e}")
            pred_i = np.nan
        resid[i] = (pred_i - H[i]) / sigma[i]
    return float(np.sqrt(np.nanmean(resid ** 2))), resid


def run_config(label, z, H, sigma, zgrid):
    print(f"\n--- {label}  (N={len(z)}) ---")
    rows, residuals = [], {}
    for name, fitter in METHODS.items():
        print(f"  Running LOOCV for {name} ...")
        score, resid = loocv(fitter, name, z, H, sigma)
        residuals[name] = resid
        chi_obs, ci_lo, ci_hi = bootstrap_chi_ci(resid)
        if name == "NN":
            predict_full = fitter(z, H, sigma, seed=0)
        else:
            predict_full = fitter(z, H, sigma)
        q = deceleration_parameter(predict_full, zgrid)
        zt_hat = transition_redshift_from_curve(zgrid, q)
        rows.append(dict(config=label, method=name, loocv_chi=score,
                          loocv_chi_ci_lo=ci_lo, loocv_chi_ci_hi=ci_hi, zt_hat=zt_hat))
        print(f"    {name:12s}  LOOCV chi={score:.3f} [{ci_lo:.3f},{ci_hi:.3f}]  zt_hat={zt_hat:.3f}")

    popt, pcov = curve_fit(flat_lcdm_model, z, H, p0=[70, 0.3], sigma=sigma,
                            absolute_sigma=True, bounds=([40, 0.05], [100, 0.95]))
    resid = np.zeros(len(z))
    for i in range(len(z)):
        mask = np.ones(len(z), dtype=bool); mask[i] = False
        try:
            p_i, _ = curve_fit(flat_lcdm_model, z[mask], H[mask], p0=[70, 0.3],
                                sigma=sigma[mask], absolute_sigma=True,
                                bounds=([40, 0.05], [100, 0.95]))
            pred_i = flat_lcdm_model(z[i], *p_i)
        except Exception:
            pred_i = np.nan
        resid[i] = (pred_i - H[i]) / sigma[i]
    residuals["ParametricLCDM"] = resid
    param_chi = float(np.sqrt(np.nanmean(resid ** 2)))
    p_ci_lo, p_ci_hi = bootstrap_chi_ci(resid)[1:]
    param_predict = lambda zq: flat_lcdm_model(np.atleast_1d(zq), *popt)
    zt_param = transition_redshift_from_curve(zgrid, deceleration_parameter(param_predict, zgrid))
    rows.append(dict(config=label, method="ParametricLCDM", loocv_chi=param_chi,
                      loocv_chi_ci_lo=p_ci_lo, loocv_chi_ci_hi=p_ci_hi, zt_hat=zt_param))
    print(f"    {'Parametric':12s}  LOOCV chi={param_chi:.3f} [{p_ci_lo:.3f},{p_ci_hi:.3f}]  "
          f"zt_hat={zt_param:.3f}  (H0={popt[0]:.2f}+/-{np.sqrt(pcov[0,0]):.2f}, "
          f"Om={popt[1]:.3f}+/-{np.sqrt(pcov[1,1]):.2f})")

    non_param = [r for r in rows if r["method"] != "ParametricLCDM"]
    ranked = sorted(non_param, key=lambda r: r["loocv_chi"])
    best, second = ranked[0]["method"], ranked[1]["method"]
    diff, pval = paired_bootstrap_pvalue(residuals[best], residuals[second])
    sig = "NOT significant" if pval >= 0.05 else "significant"
    print(f"  Paired test {best} vs {second}: diff={diff:.3f}, p={pval:.3f} ({sig} at p<0.05)")
    return rows, dict(config=label, best=best, second=second, diff=diff, pval=pval, sig=sig)


def run_real_data_torch():
    print("\n" + "=" * 70)
    print("B/C. Real-data LOOCV with the REAL torch NN")
    print("=" * 70)
    z_cc, H_cc, sig_cc = load_cc_data()
    z_bao, H_bao, sig_bao, _ = load_bao_data()

    all_rows, sig_rows = [], []
    r, s = run_config(f"CC only (N={len(z_cc)})", z_cc, H_cc, sig_cc, np.linspace(0.07, 1.965, 400))
    all_rows += r; sig_rows.append(s)
    r, s = run_config(f"BAO only (N={len(z_bao)})", z_bao, H_bao, sig_bao,
                       np.linspace(z_bao.min(), z_bao.max(), 400))
    all_rows += r; sig_rows.append(s)

    z_c = np.concatenate([z_cc, z_bao]); H_c = np.concatenate([H_cc, H_bao]); s_c = np.concatenate([sig_cc, sig_bao])
    order = np.argsort(z_c); z_c, H_c, s_c = z_c[order], H_c[order], s_c[order]
    r, s = run_config(f"CC + BAO combined (N={len(z_c)})", z_c, H_c, s_c, np.linspace(0.07, 2.36, 400))
    all_rows += r; sig_rows.append(s)

    pd.DataFrame(all_rows).to_csv("real_data_bao_combined_loocv_torch.csv", index=False)
    pd.DataFrame(sig_rows).to_csv("real_data_bao_significance_torch.csv", index=False)

    # CC-only alone, matching manuscript Table (also written separately for convenience)
    cc_rows = [r for r in all_rows if r["config"].startswith("CC only")]
    pd.DataFrame(cc_rows).to_csv("real_data_loocv_torch.csv", index=False)
    print("\nSaved: real_data_loocv_torch.csv, real_data_bao_combined_loocv_torch.csv, "
          "real_data_bao_significance_torch.csv")


# ── D. GLS covariance-aware NN with a torch precision-weighted loss ─────
def fit_nn_gls_torch(z, H, Cov, epochs=4000, lr=1e-2, hidden=64, weight_decay=1e-4, seed=0):
    torch.manual_seed(42)
    z_mean, z_std = z.mean(), z.std()
    H_mean, H_std = H.mean(), H.std()
    zt = torch.tensor(((z - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
    Ht = torch.tensor(((H - H_mean) / H_std).reshape(-1, 1), dtype=torch.float32)
    Cinv = torch.tensor(np.linalg.inv(Cov) * (H_std ** 2), dtype=torch.float32)

    model = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                           nn.Linear(hidden, hidden), nn.Tanh(),
                           nn.Linear(hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(z)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(zt)
        r = (pred - Ht).flatten()
        loss = (r @ Cinv @ r) / n
        loss.backward()
        opt.step()

    def predict(zq):
        zq = np.atleast_1d(zq).astype(float)
        with torch.no_grad():
            zqt = torch.tensor(((zq - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
            out = model(zqt).numpy().flatten()
        return out * H_std + H_mean

    return predict


def run_gls_torch():
    print("\n" + "=" * 70)
    print("D. GLS covariance-aware NN fit with real torch (Sec. 5.4 of manuscript)")
    print("=" * 70)
    from covariance import build_covariance, fit_poly_gls, fit_gp_gls, diag_only_weights
    from methods import fit_spline, fit_nodal
    z, H, sigma_stat = load_cc_data()
    Cov = build_covariance(z, H, sigma_stat)

    def loocv_gls(fit_fn):
        n = len(z)
        resid = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            Cov_sub = Cov[np.ix_(mask, mask)]
            try:
                predict = fit_fn(z[mask], H[mask], Cov_sub)
                pred_i = predict(z[i])
                pred_i = pred_i[0] if hasattr(pred_i, "__len__") else pred_i
            except Exception:
                pred_i = np.nan
            resid[i] = (pred_i - H[i]) / np.sqrt(Cov[i, i])
        return float(np.sqrt(np.nanmean(resid ** 2))), resid

    rows = []
    for name, fn in [("Polynomial (GLS)", fit_poly_gls), ("GP (GLS)", fit_gp_gls),
                      ("NN (GLS, torch)", fit_nn_gls_torch)]:
        score, resid = loocv_gls(fn)
        chi_obs, ci_lo, ci_hi = bootstrap_chi_ci(resid)
        predict_full = fn(z, H, Cov)
        q = deceleration_parameter(predict_full, np.linspace(0.07, 1.965, 400))
        zt_hat = transition_redshift_from_curve(np.linspace(0.07, 1.965, 400), q)
        rows.append(dict(method=name, loocv_chi=score, loocv_chi_ci_lo=ci_lo,
                          loocv_chi_ci_hi=ci_hi, zt_hat=zt_hat, weighting="full covariance (GLS)"))
        print(f"  {name}: LOOCV chi={score:.3f} [{ci_lo:.3f},{ci_hi:.3f}], zt={zt_hat:.3f}")

    sigma_full = diag_only_weights(Cov)
    for name, fitter in [("Spline (diag-only)", fit_spline), ("Nodal (diag-only)", fit_nodal)]:
        n = len(z)
        resid = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            try:
                predict = fitter(z[mask], H[mask], sigma_full[mask])
                pred_i = predict(z[i])[0]
            except Exception:
                pred_i = np.nan
            resid[i] = (pred_i - H[i]) / sigma_full[i]
        score = float(np.sqrt(np.nanmean(resid ** 2)))
        chi_obs, ci_lo, ci_hi = bootstrap_chi_ci(resid)
        predict_full = fitter(z, H, sigma_full)
        q = deceleration_parameter(predict_full, np.linspace(0.07, 1.965, 400))
        zt_hat = transition_redshift_from_curve(np.linspace(0.07, 1.965, 400), q)
        rows.append(dict(method=name, loocv_chi=score, loocv_chi_ci_lo=ci_lo,
                          loocv_chi_ci_hi=ci_hi, zt_hat=zt_hat,
                          weighting="diagonal only (stat+sys in quadrature)"))
        print(f"  {name}: LOOCV chi={score:.3f} [{ci_lo:.3f},{ci_hi:.3f}], zt={zt_hat:.3f}")

    df = pd.DataFrame(rows).sort_values("loocv_chi")
    df.to_csv("real_data_covariance_loocv_torch.csv", index=False)
    print("\nSaved: real_data_covariance_loocv_torch.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_consistency_check()
    run_real_data_torch()
    run_gls_torch()
    print("\n\nDONE. Please send back:")
    print("  - the full console output above")
    print("  - torch_vs_numpy_consistency.csv")
    print("  - real_data_loocv_torch.csv")
    print("  - real_data_bao_combined_loocv_torch.csv")
    print("  - real_data_bao_significance_torch.csv")
    print("  - real_data_covariance_loocv_torch.csv")
