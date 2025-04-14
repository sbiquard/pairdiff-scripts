#!/usr/bin/env python3

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedFormatter, FixedLocator
from sigma_to_epsilon import get_epsilon_samples, get_scatters

sns.color_palette()

SAVE_PLOTS_DIR = Path(__file__).parents[1] / "jz_out" / "analysis" / "optimality"
SAVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCATTERS = [0.001, 0.01, 0.1, 0.2]


def my_savefig(fig, title: str, close: bool = True, dpi=200):
    fig.savefig(SAVE_PLOTS_DIR / title, bbox_inches="tight", dpi=dpi)
    if close:
        plt.close(fig)


# sample scatter values
scatters = get_scatters((0.9 * SCATTERS[0], 1.1 * SCATTERS[-1]))
eps = get_epsilon_samples(jax.random.key(1426), scatters)
alpha = 1 / (1 - eps**2)

# loop over hwp and no_hwp cases
for k_hwp in ["hwp", "no_hwp"]:
    # load data
    data = np.load(SAVE_PLOTS_DIR / "data" / f"variance_increase_white_{k_hwp}.npz")
    x_m = np.array(SCATTERS)
    q = np.array([10, 90, 1, 99])

    fig, ax = plt.subplots()
    hwp_title = k_hwp.replace("_", " ")
    ax.set(
        xlabel="Scatter around nominal NET",
        ylabel="Variance increase",
        title=f"Variance increase per pixel ({hwp_title})",
    )

    # Expected variance increase as a function of scatter
    x = scatters
    y = alpha.mean(axis=-1) - 1
    ax.plot(x, y, "k", label="expected increase")

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
    err_q = np.abs([[pq[0], pq[1]] for pq in data["pct_q"]]).T
    err_u = np.abs([[pu[0], pu[1]] for pu in data["pct_u"]]).T

    # err_q = data['std_q']
    # err_u = data['std_u']

    ax.errorbar(x_m, data["avg_q"], yerr=err_q, ls="", marker=5, capsize=3, label="Q increase")
    ax.errorbar(x_m, data["avg_u"], yerr=err_u, ls="", marker=4, capsize=3, label="U increase")

    ax.set_xscale("log")
    ax.set_yscale("asinh", linear_width=1e-4)
    ax.set_ylim(-0.03 / 100, 0.25)  # -0.3 to 25 percent

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

    my_savefig(fig, f"variance_increase_scatter_{k_hwp}")
