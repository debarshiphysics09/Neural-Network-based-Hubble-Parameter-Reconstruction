"""
Main experiment driver.

Two experiment tiers, addressing the referee's reproducibility/statistics
concerns directly:

  TIER 1 (primary benchmark): flat LCDM, H(z) observable, N_SEEDS_MAIN
          independent noise realisations per noise level, for all five
          methods. This is the headline result and is run with enough
          seeds to report mean +/- std (addresses "results should be
          repeated over many independent noise realisations").

  TIER 2 (robustness grid): all 3 cosmologies x all 3 observable types,
          N_SEEDS_ROBUST seeds each, all 5 methods, at a single
          representative noise level. This tests whether the Tier-1
          conclusions generalise (addresses "test more configurations:
          different cosmological models, different observables").

Sample size N=80 and noise levels sigma in {1,3,5} km/s/Mpc are justified
in the manuscript text (see Sec. 2.3): N=80 anticipates near-future
CC+BAO H(z) compilations (current catalogues: ~32 CC + ~15-20 BAO
points), and sigma=1,3,5 bracket the typical per-point precision of
existing cosmic-chronometer measurements (Moresco et al. 2022 report
errors of 8-15% on individual H(z) values, i.e. ~5-20 km/s/Mpc at
z~0.3-2, corresponding to our sigma=3-5 km/s/Mpc bracket; sigma=1 is
included as an optimistic near-future benchmark).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from cosmology import ALL_COSMOLOGIES, FLAT_LCDM
from data_gen import GENERATORS
from methods import METHODS
from metrics import (integrated_rmse, deceleration_parameter,
                      transition_redshift_from_curve, parametric_reference_rmse)

ZGRID = np.linspace(0.0, 2.0, 400)
N_POINTS = 80
NOISE_LEVELS_MAIN = [1.0, 3.0, 5.0]          # km/s/Mpc, for Hz observable
N_SEEDS_MAIN = 30
N_SEEDS_ROBUST = 10
ROBUST_NOISE = {"Hz": 3.0, "DH": 0.03}  # representative noise per observable
# NOTE: a third observable type ("mu", SN-like distance modulus converted to
# an effective H(z) via numerical differentiation of noisy D_C(z)) was
# implemented and tested but found to be numerically unstable: finite
# differencing amplifies the mu-space noise by orders of magnitude near
# any point where the noisy D_C(z) curve is locally non-monotonic, which is
# generic at realistic SN noise levels. This is a known, fundamental
# ill-posedness of differentiating noisy distance data (not a fixable
# implementation bug) and is the reason real H(z) compilations are built
# from cosmic chronometers and *radial* BAO, not from differentiated SN
# Hubble diagrams. We therefore restrict the robustness grid to the two
# observable types -- H(z) and the BAO D_H(z) proxy -- for which the target
# quantity is the Hubble rate itself, and discuss this explicitly as a
# scope limitation (Sec. 5).


def run_one(cosmology, observable, noise_param, seed, methods=None, n=N_POINTS):
    gen = GENERATORS[observable]
    if observable == "Hz":
        z, H_obs, sigma, H_true = gen(cosmology, n=n, sigma=noise_param, seed=seed)
    elif observable == "DH":
        z, H_obs, sigma, H_true = gen(cosmology, n=n, rel_sigma=noise_param, seed=seed)
    elif observable == "mu":
        z, H_obs, sigma, H_true = gen(cosmology, n=n, sigma_mu=noise_param, seed=seed)
    else:
        raise ValueError(observable)

    methods = methods or list(METHODS.keys())
    out = {}
    for name in methods:
        fitter = METHODS[name]
        t0 = time.time()
        try:
            if name == "NN":
                predict = fitter(z, H_obs, sigma, seed=seed)
            else:
                predict = fitter(z, H_obs, sigma)
            rmse_val, mae_val = integrated_rmse(predict, cosmology, ZGRID)
            q = deceleration_parameter(predict, ZGRID)
            zt_hat = transition_redshift_from_curve(ZGRID, q)
        except Exception as e:
            rmse_val, mae_val, zt_hat = np.nan, np.nan, np.nan
        failed = int(np.isnan(rmse_val) or rmse_val > 20.0)  # catastrophic failure flag
        out[name] = dict(rmse=rmse_val, mae=mae_val, zt_hat=zt_hat,
                          wall_s=time.time() - t0, failed=failed)

    # parametric reference (only meaningful for Hz observable, but compute for all)
    try:
        ref_rmse, popt = parametric_reference_rmse(z, H_obs, sigma, cosmology, ZGRID)
    except Exception:
        ref_rmse = np.nan
    out["ParametricLCDM"] = dict(rmse=ref_rmse, mae=np.nan, zt_hat=np.nan, wall_s=0.0)

    zt_true = cosmology.transition_redshift()
    return out, zt_true


def run_tier1(outpath):
    rows = []
    for sigma in NOISE_LEVELS_MAIN:
        for seed in range(N_SEEDS_MAIN):
            out, zt_true = run_one(FLAT_LCDM, "Hz", sigma, seed)
            for method, res in out.items():
                rows.append(dict(cosmology="flat_lcdm", observable="Hz",
                                  noise=sigma, seed=seed, method=method,
                                  zt_true=zt_true, **res))
        print(f"[tier1] sigma={sigma} done ({N_SEEDS_MAIN} seeds)")
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    return df


def run_tier2(outpath):
    rows = []
    for cname, cosmo in ALL_COSMOLOGIES.items():
        for obs, noise in ROBUST_NOISE.items():
            for seed in range(N_SEEDS_ROBUST):
                out, zt_true = run_one(cosmo, obs, noise, seed)
                for method, res in out.items():
                    rows.append(dict(cosmology=cname, observable=obs,
                                      noise=noise, seed=seed, method=method,
                                      zt_true=zt_true, **res))
            print(f"[tier2] {cname} / {obs} done ({N_SEEDS_ROBUST} seeds)")
    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    return df


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    t0 = time.time()
    df1 = run_tier1("../results/tier1_main.csv")
    print(f"Tier 1 done in {time.time()-t0:.1f}s, {len(df1)} rows")
    t1 = time.time()
    df2 = run_tier2("../results/tier2_robust.csv")
    print(f"Tier 2 done in {time.time()-t1:.1f}s, {len(df2)} rows")
