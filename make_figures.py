import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
    "legend.fontsize": 10.5, "figure.dpi": 140,
})

from cosmology import FLAT_LCDM
from data_gen import generate_Hz_data
from methods import METHODS
from metrics import deceleration_parameter, transition_redshift_from_curve

FIGDIR = "../figures"
os.makedirs(FIGDIR, exist_ok=True)
COLORS = {"NN": "#16325c", "Polynomial": "#e07b1a", "Spline": "#7b3fa0",
          "GP": "#1a9e6e", "Nodal": "#c0392b"}

ZGRID = np.linspace(0, 2, 400)

# ── Fig 1: example reconstructions for all 5 methods, sigma=3, one seed ─────
def fig_reconstructions():
    z, H_obs, sigma, H_true = generate_Hz_data(FLAT_LCDM, n=80, sigma=3.0, seed=0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()
    for ax, (name, fitter) in zip(axes, METHODS.items()):
        predict = fitter(z, H_obs, sigma, seed=0) if name == "NN" else fitter(z, H_obs, sigma)
        pred = predict(ZGRID)
        true = FLAT_LCDM.Hz(ZGRID).flatten()
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        ax.scatter(z, H_obs, s=14, color="lightcoral", alpha=0.6, label="Mock data", zorder=2)
        ax.plot(ZGRID, true, "k--", lw=1.8, label="True $H(z)$")
        ax.plot(ZGRID, pred, color=COLORS[name], lw=2, label=name)
        ax.set_title(f"{name}  (RMSE={rmse:.2f})")
        ax.set_xlabel("$z$"); ax.set_ylabel("$H(z)$")
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].axis("off")
    fig.suptitle(r"Reconstructions of $H(z)$, all five methods ($\sigma=3\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$, single realisation)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig1_all_methods_reconstruction.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 2: RMSE vs noise, mean +/- std over 30 seeds (Tier 1) ───────────────
def fig_rmse_vs_noise():
    df = pd.read_csv("../results/tier1_main.csv")
    df = df[df.method != "ParametricLCDM"]
    fig, ax = plt.subplots(figsize=(7, 5.3))
    for name in METHODS:
        g = df[df.method == name].groupby("noise")["rmse"].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], "-o", color=COLORS[name], label=name, lw=2)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                         color=COLORS[name], alpha=0.15)
    ax.set_xlabel(r"Noise level $\sigma$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"RMSE  [km s$^{-1}$ Mpc$^{-1}$]  (mean $\pm$ std, 30 seeds)")
    ax.set_title("Integrated RMSE vs noise level (flat $\\Lambda$CDM, $H(z)$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig2_rmse_vs_noise_multiseed.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 3: RMSE vs reference (parametric LCDM + nonparametric methods) ──────
def fig_parametric_comparison():
    df = pd.read_csv("../results/tier1_main.csv")
    fig, ax = plt.subplots(figsize=(7, 5.3))
    for name in list(METHODS.keys()) + ["ParametricLCDM"]:
        g = df[df.method == name].groupby("noise")["rmse"].mean()
        ls = "k--" if name == "ParametricLCDM" else "-o"
        col = "black" if name == "ParametricLCDM" else COLORS[name]
        lab = "Correctly-specified parametric fit (2 free params)" if name == "ParametricLCDM" else name
        ax.plot(g.index, g.values, ls, color=col, label=lab, lw=2.2 if name=="ParametricLCDM" else 2)
    ax.set_xlabel(r"Noise level $\sigma$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel(r"RMSE  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("Nonparametric methods vs. the correctly-specified\nparametric reference (model-independence cost)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig3_parametric_reference.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 4: transition redshift recovery, boxplot-style over seeds ───────────
def fig_zt_recovery():
    df = pd.read_csv("../results/tier1_main.csv")
    df = df[df.method != "ParametricLCDM"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, sigma in zip(axes, [1.0, 3.0, 5.0]):
        sub = df[df.noise == sigma]
        data = [sub[sub.method == m]["zt_hat"].dropna().values for m in METHODS]
        bp = ax.boxplot(data, labels=list(METHODS.keys()), patch_artist=True, widths=0.6)
        for patch, name in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[name]); patch.set_alpha(0.5)
        ax.axhline(FLAT_LCDM.transition_redshift(), color="k", ls="--", lw=1.5, label="True $z_t$")
        ax.set_title(f"$\\sigma={sigma:.0f}$")
        ax.tick_params(axis="x", rotation=30)
        if sigma == 1.0:
            ax.set_ylabel("$\\hat{z}_t$")
        ax.legend(fontsize=8)
    fig.suptitle("Recovered transition redshift $\\hat{z}_t$ across 30 noise realisations", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig4_zt_recovery_boxplot.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 5: robustness grid heatmap across cosmologies x observables ─────────
def fig_robustness_grid():
    df = pd.read_csv("../results/tier2_robust.csv")
    df = df[df.method != "ParametricLCDM"]
    cosmo_labels = {"flat_lcdm": "Flat $\\Lambda$CDM", "nonflat_lcdm": "Non-flat $\\Lambda$CDM",
                     "cpl_de": "CPL dynamical DE"}
    obs_labels = {"Hz": "$H(z)$ (CC)", "DH": "BAO $D_H(z)$"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    width = 0.15
    methods = list(METHODS.keys())
    for ax, obs in zip(axes, ["Hz", "DH"]):
        x = np.arange(3)
        for i, name in enumerate(methods):
            vals = []
            for c in ["flat_lcdm", "nonflat_lcdm", "cpl_de"]:
                sub = df[(df.cosmology == c) & (df.observable == obs) & (df.method == name)]
                vals.append(sub["rmse"].mean())
            ax.bar(x + i * width, vals, width=width, color=COLORS[name], label=name)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([cosmo_labels[c] for c in ["flat_lcdm", "nonflat_lcdm", "cpl_de"]],
                            rotation=10, fontsize=9)
        ax.set_title(obs_labels[obs])
        ax.set_ylabel("Mean RMSE [km s$^{-1}$ Mpc$^{-1}$]" if obs == "Hz" else "")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Robustness grid: mean RMSE across 3 cosmologies $\\times$ 2 observables (10 seeds each)", y=1.01, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{FIGDIR}/fig5_robustness_grid.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 6: q(z) reconstruction, multi-method, sigma=3 ────────────────────────
def fig_qz():
    z, H_obs, sigma, H_true = generate_Hz_data(FLAT_LCDM, n=80, sigma=3.0, seed=0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    qtrue = FLAT_LCDM.q(ZGRID)
    ax.plot(ZGRID, qtrue, "k--", lw=2, label="True $q(z)$")
    zt_true = FLAT_LCDM.transition_redshift()
    ax.axvline(zt_true, color="gray", ls=":", lw=1.3)
    for name, fitter in METHODS.items():
        predict = fitter(z, H_obs, sigma, seed=0) if name == "NN" else fitter(z, H_obs, sigma)
        q = deceleration_parameter(predict, ZGRID)
        ax.plot(ZGRID, q, color=COLORS[name], lw=1.8, label=name, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("$z$"); ax.set_ylabel("$q(z)$")
    ax.set_title(r"Deceleration parameter $q(z)$ from reconstructed $H(z)$ ($\sigma=3$)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig6_qz_all_methods.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 7: RMSE vs parametric-reference RMSE ratio (model-independence cost) ─
def fig_efficiency_ratio():
    df = pd.read_csv("../results/tier1_main.csv")
    piv = df.pivot_table(index=["noise", "seed"], columns="method", values="rmse")
    fig, ax = plt.subplots(figsize=(7, 5.3))
    for name in METHODS:
        ratio = piv[name] / piv["ParametricLCDM"]
        g = ratio.groupby("noise").agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt="-o", color=COLORS[name],
                    label=name, capsize=3)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="Parametric reference (ratio=1)")
    ax.set_xlabel(r"Noise level $\sigma$  [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel("RMSE / RMSE(correctly-specified parametric fit)")
    ax.set_title("Cost of model-independence")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig7_model_independence_cost.png", bbox_inches="tight")
    plt.close(fig)


# ── Fig 8: NN training loss (regenerate from a representative run) ──────────
def fig_training_loss():
    import torch
    from methods import HzNet
    fig, ax = plt.subplots(figsize=(7, 5))
    for sigma, color in zip([1, 3, 5], ["#2ca02c", "#ff7f0e", "#d62728"]):
        z, H, sig, _ = generate_Hz_data(FLAT_LCDM, n=80, sigma=float(sigma), seed=0)
        torch.manual_seed(0)
        z_mean, z_std = z.mean(), z.std(); H_mean, H_std = H.mean(), H.std()
        zt = torch.tensor(((z - z_mean) / z_std).reshape(-1, 1), dtype=torch.float32)
        Ht = torch.tensor(((H - H_mean) / H_std).reshape(-1, 1), dtype=torch.float32)
        w = torch.tensor((1.0 / sig ** 2).reshape(-1, 1), dtype=torch.float32); w = w / w.mean()
        model = HzNet(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for _ in range(2000):
            opt.zero_grad(); pred = model(zt); loss = (w * (pred - Ht) ** 2).mean()
            loss.backward(); opt.step(); losses.append(loss.item())
        ax.plot(losses, color=color, label=f"$\\sigma={sigma}$")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MSE loss (normalised units)")
    ax.set_title("Neural network training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig8_training_loss.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_reconstructions(); print("fig1 done")
    fig_rmse_vs_noise(); print("fig2 done")
    fig_parametric_comparison(); print("fig3 done")
    fig_zt_recovery(); print("fig4 done")
    fig_robustness_grid(); print("fig5 done")
    fig_qz(); print("fig6 done")
    fig_efficiency_ratio(); print("fig7 done")
    fig_training_loss(); print("fig8 done")
