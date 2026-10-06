"""
REFEREE FIX: bootstrap confidence intervals and paired significance
testing for the LOOCV comparisons in run_real_data.py / run_bao_and_combined.py.

Without this, differences of 0.02-0.2 in LOOCV chi across methods on
N=10-31 points cannot be distinguished from noise, and the manuscript's
claim that the real-data ranking "reorders" relative to the synthetic
ranking is unsupported. We estimate:

  1. A bootstrap CI on each method's LOOCV chi, by resampling the N
     per-point standardised LOOCV residuals (with replacement) and
     recomputing chi, B=2000 times.
  2. A paired bootstrap test for whether the TOP TWO methods in a given
     configuration are significantly different: for each bootstrap
     resample (same resampled indices for both methods, preserving
     pairing), compute chi_A - chi_B; report the fraction of resamples
     with the opposite sign to the observed difference as a two-sided
     p-value proxy.

This requires access to the per-point LOOCV residuals, not just the
aggregate chi, so the driver scripts are extended to save residuals.
"""
import numpy as np


def bootstrap_chi_ci(resid, B=2000, seed=0, ci=(16, 84)):
    """resid: array of standardised LOOCV residuals (pred-obs)/sigma.
    Returns (chi_obs, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    resid = resid[~np.isnan(resid)]
    n = len(resid)
    chi_obs = float(np.sqrt(np.mean(resid ** 2)))
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = np.sqrt(np.mean(resid[idx] ** 2))
    lo, hi = np.percentile(boots, ci)
    return chi_obs, float(lo), float(hi)


def paired_bootstrap_pvalue(resid_a, resid_b, B=2000, seed=0):
    """Two-sided bootstrap p-value for chi_a != chi_b, using the SAME
    resampled indices for both (paired resampling, since both methods
    are evaluated on the same held-out points)."""
    rng = np.random.default_rng(seed)
    resid_a = np.asarray(resid_a); resid_b = np.asarray(resid_b)
    mask = ~np.isnan(resid_a) & ~np.isnan(resid_b)
    resid_a, resid_b = resid_a[mask], resid_b[mask]
    n = len(resid_a)
    obs_diff = np.sqrt(np.mean(resid_a ** 2)) - np.sqrt(np.mean(resid_b ** 2))
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = np.sqrt(np.mean(resid_a[idx] ** 2)) - np.sqrt(np.mean(resid_b[idx] ** 2))
    # two-sided: fraction of bootstrap diffs at least as far from 0, in
    # the direction opposite the observed sign, doubled
    if obs_diff >= 0:
        p = 2 * np.mean(boots <= 0)
    else:
        p = 2 * np.mean(boots >= 0)
    return float(obs_diff), float(min(p, 1.0))
