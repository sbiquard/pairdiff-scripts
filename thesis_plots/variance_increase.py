#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedFormatter, FixedLocator, PercentFormatter
from sigma_to_epsilon import get_epsilon_samples, get_scatters

sns.set_theme(context="paper", style="ticks")
default_palette = sns.color_palette()

JZ_OUT_DIR = Path(__file__).parents[1] / "jz_out"
SAVE_PLOTS_DIR = JZ_OUT_DIR / "analysis" / "optimality"
SAVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCATTERS = [0.001, 0.01, 0.1, 0.2]


def my_savefig(fig, title: str, close: bool = True, dpi=600):
    fig.savefig(title, bbox_inches="tight", dpi=dpi)
    if close:
        plt.close(fig)


LONRA = [-95, 135]
LATRA = [-70, -10]

cartview_sky = partial(hp.cartview, lonra=LONRA, latra=LATRA, xsize=5000)


# epsilon distributions that were used in the simulation
epsilons = {
    k_ml_pd: {
        k_hwp: [
            np.load(
                JZ_OUT_DIR
                / "opti"
                / ("var_increase" + (f"_{k_hwp}" if k_hwp == "no_hwp" else ""))
                / f"scatter_{scatter}"
                / k_ml_pd
                / "epsilon_dist.npy"
            )
            for scatter in SCATTERS
        ]
        for k_hwp in ["hwp", "no_hwp"]
    }
    for k_ml_pd in ["ml", "pd"]
}

eta_maps = {
    k_hwp: [
        np.load(JZ_OUT_DIR / "analysis/optimality/data" / f"eta_scatter{scatter}_{k_hwp}.npz")
        for scatter in SCATTERS
    ]
    for k_hwp in ["hwp", "no_hwp"]
}

# sample scatter values
scatters = get_scatters((0.9 * SCATTERS[0], 1.1 * SCATTERS[-1]))
eps = get_epsilon_samples(jax.random.key(1426), scatters)
eta = 1 / (1 - eps**2)
eta_rel = eta.mean(axis=-1) - 1

fhist, axsh = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
fmaps = {k: plt.figure(figsize=(10, 4)) for k in eta_maps}
for k in eta_maps:
    for i, scatter in enumerate(SCATTERS):
        eta = eta_maps[k][i]
        qq = np.where(eta["qq"] < 0.5, np.nan, eta["qq"])  # unphysical
        uu = np.where(eta["uu"] < 0.5, np.nan, eta["uu"])  # unphysical
        # plot the eta maps for Q and U
        plt.figure(fmaps[k])
        cartview_sky((qq - 1) * 100, sub=221 + i, title=f"z = {scatter:.1%}", unit=r"$\%$")
        # cartview_sky(eta["uu"], sub=221 + i)
        # histogram
        ax = axsh.flat[i]

        # Calculate histogram and bin centers for Q values
        q_hist, q_bins = np.histogram((qq - 1)[~np.isnan(qq)], bins=100)
        q_bin_centers = (q_bins[:-1] + q_bins[1:]) / 2

        # Plot histogram
        stair = ax.stairs(q_hist, q_bins, label=f"Q {k}" if i == 0 else None)

        # Fit Gaussian to the central part of the histogram data
        # Define the central region (e.g., within 1.5 standard deviations)
        q_values = (qq - 1)[~np.isnan(qq - 1)]
        q_mean = np.nanmean(q_values)
        q_std = np.nanstd(q_values)

        # Filter data to use only central part for fitting
        central_mask = np.abs(q_values - q_mean) < 1.5 * q_std
        central_values = q_values[central_mask]

        # Recalculate mean and std using only central data
        q_mean = np.mean(central_values)
        q_std = np.std(central_values)

        # Generate the Gaussian curve
        q_gaussian = np.exp(-0.5 * ((q_bin_centers - q_mean) / q_std) ** 2) / (
            q_std * np.sqrt(2 * np.pi)
        )
        q_gaussian = (
            q_gaussian * np.max(q_hist) / np.max(q_gaussian)
        )  # Scale to match histogram height

        # Plot the Gaussian fit
        ax.plot(q_bin_centers, q_gaussian, ls="--", lw=0.8, color="b" if k == "hwp" else "g")
        # ax.axvline(np.nanmean(qq) - 1, color="k", ls="--")

        # Add shaded region for Q hwp cases between 10th and 90th percentiles
        if k == "hwp":
            # Calculate 10th and 90th percentiles
            p01 = np.percentile(q_values, 1)
            p99 = np.percentile(q_values, 99)

            # Create shaded region
            ax.axvspan(
                p01,
                p99,
                alpha=0.3,
                color="grey",
                label="1-99th percentile (Q-hwp)" if i == 0 else None,
            )

        # U histogram
        # Calculate histogram and bin centers for U values
        u_hist, u_bins = np.histogram((uu - 1)[~np.isnan(uu)], bins=100)
        u_bin_centers = (u_bins[:-1] + u_bins[1:]) / 2

        # Plot histogram using stairs
        ax.stairs(u_hist, u_bins, label=f"U {k}" if i == 0 else None)
fhist.legend(loc="outside upper center", ncols=5)
for i, ax in enumerate(axsh.flat):
    ax.axvline(eta_rel[np.argmin(np.abs(scatters - SCATTERS[i]))], color="k", ls="--")
    ax.set_yscale("log")
    ax.set_ylim(1, 1e5)
    if i >= 2:
        ax.set_xlabel("Relative variance increase")
    if i % 2 == 0:
        ax.set_ylabel("Count")
    if i < 2:
        dec = 2
    elif i < 3:
        dec = 1
    else:
        dec = 0
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=dec))
    ax.text(
        0.85,
        0.95,
        f"$z$ = {SCATTERS[i]:.1%}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
# f.tight_layout()
my_savefig(fhist, "eta_histograms.svg")
for k in fmaps:
    my_savefig(fmaps[k], title=f"eta_maps_Q_{k}.svg")

# for i, _ in enumerate(SCATTERS):
#     np.testing.assert_allclose(epsilons["ml"]["hwp"][i], epsilons["pd"]["hwp"][i])

sns.set_palette(default_palette)

# loop over hwp and no_hwp cases
for k_hwp in ["hwp", "no_hwp"]:
    # load data
    data = np.load(SAVE_PLOTS_DIR / "data" / f"variance_increase_white_{k_hwp}.npz")
    x_m = np.array(SCATTERS)
    q = np.array([10, 90, 1, 99])
    eps_m = np.stack([epsilons["ml"][k_hwp][i] for i in range(len(SCATTERS))])
    eta_m = (1 / (1 - eps_m**2)).mean(axis=-1)

    fig, ax = plt.subplots()
    hwp_title = k_hwp.replace("_", " ")
    ax.set(
        xlabel="Scatter around nominal NET",
        ylabel="Relative variance increase",
        # title=f"Variance increase per pixel ({hwp_title})",
    )

    # Expected variance increase as a function of scatter
    x = scatters
    ax.plot(x, eta_rel, "k", label="true expectation")
    ax.scatter(x_m, eta_m - 1, s=128, marker="x", c="r", label="empirical expectation")
    # ax.scatter(x_m, 2 * (eta_m - 1), s=128, marker="x", c="g", label="2x empirical expectation")

    # # interpolate measured percentiles in log space
    # y_10 = jnp.interp(jnp.log(x), jnp.log(x_m), jnp.log(jnp.array([_p[0] for _p in p_qq])))
    # y_90 = jnp.interp(jnp.log(x), jnp.log(x_m), jnp.log(jnp.array([_p[1] for _p in p_qq])))
    # y_01 = jnp.interp(jnp.log(x), jnp.log(x_m), jnp.log(jnp.array([_p[2] for _p in p_qq])))
    # y_99 = jnp.interp(jnp.log(x), jnp.log(x_m), jnp.log(jnp.array([_p[3] for _p in p_qq])))

    # ax.fill_between(
    #     x, jnp.exp(y_10), jnp.exp(y_90), alpha=0.5, color="g", label="10-90th percentile"
    # )
    # ax.fill_between(
    #     x, jnp.exp(y_01), jnp.exp(y_99), alpha=0.5, color="orange", label="1-99th percentile"
    # )

    # ax.scatter(x_m, means_qq, marker=5, color="r", label="QQ average")
    # ax.scatter(x_m, means_uu, marker=4, color="r", label="UU average")

    # 10-90 percentiles as error bars
    q_01_99 = np.array([[pq[2], pq[3]] for pq in data["pct_q"]]).T
    u_01_99 = np.array([[pu[2], pu[3]] for pu in data["pct_u"]]).T

    # err_q = np.zeros((2, 4))
    # err_u = np.zeros((2, 4))
    # for i in range(len(SCATTERS)):
    #     err_q[:, i] = np.nanpercentile(eta_maps[k_hwp][i]["qq"] - 1, [10, 90])
    #     err_u[:, i] = np.nanpercentile(eta_maps[k_hwp][i]["uu"] - 1, [10, 90])

    ax.errorbar(
        x_m,
        data["avg_q"],
        yerr=np.abs(q_01_99 - data["avg_q"]),
        ls="",
        marker=5,
        capsize=3,
        label="Q pixels (1-99th percentiles)",
    )
    ax.errorbar(
        x_m,
        data["avg_u"],
        yerr=np.abs(u_01_99 - data["avg_u"]),
        ls="",
        marker=4,
        capsize=3,
        label="U pixels (1-99th percentiles)",
    )

    ax.set_xscale("log")
    ax.set_yscale("asinh", linear_width=1e-4)
    ax.set_ylim(-3e-4, 0.25)  # -0.3 to 25 percent

    # Apply custom ticks to both axes
    x_tick_locations = np.array([0.1, 1, 10, 20]) / 100  # percents
    x_tick_labels = ["0.1%", "1%", "10%", "20%"]
    ax.xaxis.set_major_locator(FixedLocator(x_tick_locations))
    ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
    y_tick_locations = np.array([0, 0.1, 1, 5, 10, 20]) / 100  # percents
    y_tick_labels = ["0%", "0.1%", "1%", "5%", "10%", "20%"]
    ax.yaxis.set_major_locator(FixedLocator(y_tick_locations))
    ax.yaxis.set_major_formatter(FixedFormatter(y_tick_labels))

    ax.legend()
    ax.grid(True)

    my_savefig(fig, f"variance_increase_{k_hwp}.svg")
