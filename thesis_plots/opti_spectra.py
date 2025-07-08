#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from theory_spectra import get_theory_powers

sns.set_theme(context="notebook", style="ticks")


class NoiseType(Enum):
    """Enum representing the type of noise."""

    WHITE = "white"
    INSTR = "instr"


JZ = Path(__file__).parents[1].absolute() / "jz_out"
OPTI = JZ / "opti"
SCATTERS = [0.001, 0.01, 0.1, 0.2]
NUM_REAL = 25
MASK_REF = "hits_10000"
LMIN, LMAX = 30, 500

runs = {
    k_white: {
        k_hwp: {
            scatter: {
                k_ml_pd: [
                    OPTI
                    / (
                        "var_increase"
                        + ("_instr" if k_white != "white" else "")
                        + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")
                    )
                    / f"{real + 1:03d}"
                    / f"scatter_{scatter}"
                    # / (k_ml_pd if k_ml_pd == "ml" else "pd_new")
                    / k_ml_pd
                    for real in range(NUM_REAL)
                ]
                for k_ml_pd in ["ml", "pd"]
            }
            for scatter in SCATTERS
        }
        for k_hwp in ["hwp", "no_hwp"]
    }
    for k_white in ["white", "instr"]
}


def load_npz(path):
    data = np.load(path)
    # remove unnecessary dimensions
    return {k: data[k].squeeze() for k in data}


def cl2dl(ell, cl):
    return ell * (ell + 1) / 2 / np.pi * cl


def load_spectra(path: Path | list[Path], dl=False):
    cl = load_npz(path)
    ell = cl.pop("ell_arr")
    good = (LMIN <= ell) & (ell <= LMAX)
    if dl:
        return {"ells": ell[good], **{k: cl2dl(ell[good], cl[k][..., good]) for k in cl}}
    else:
        return {"ells": ell[good], **{k: cl[k][..., good] for k in cl}}


theory_cl = get_theory_powers()
ls = np.arange(theory_cl["BB"].size)
theory_dl = jax.tree.map(lambda x: cl2dl(ls, x), theory_cl)
lens_BB = get_theory_powers(r=0)["BB"]
prim_BB = theory_cl["BB"] - lens_BB
prim_BB_dl = cl2dl(ls, prim_BB)


input_cl = load_spectra(JZ / "input_cells_mask_apo_1000.npz")
input_dl = load_spectra(JZ / "input_cells_mask_apo_1000.npz", dl=True)
full_cl_white = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz"), runs["white"]
)
full_cl_instr = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz"), runs["instr"]
)
full_dl_white = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz", dl=True), runs["white"]
)
full_dl_instr = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz", dl=True), runs["instr"]
)
noise_cl_white = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz"), runs["white"]
)
noise_cl_instr = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz"), runs["instr"]
)
noise_dl_white = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz", dl=True), runs["white"]
)
noise_dl_instr = jax.tree.map(
    lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz", dl=True), runs["instr"]
)


def _stack(cl):
    # turn list of dicts into a single dict with stacked arrays
    return {k: np.stack(tuple(cl[i][k] for i in range(len(cl)))) for k in cl[0]}


stack_full_cl_white = jax.tree.map(_stack, full_cl_white, is_leaf=lambda x: isinstance(x, list))
stack_full_cl_instr = jax.tree.map(_stack, full_cl_instr, is_leaf=lambda x: isinstance(x, list))
stack_full_dl_white = jax.tree.map(_stack, full_dl_white, is_leaf=lambda x: isinstance(x, list))
stack_full_dl_instr = jax.tree.map(_stack, full_dl_instr, is_leaf=lambda x: isinstance(x, list))
stack_noise_cl_white = jax.tree.map(_stack, noise_cl_white, is_leaf=lambda x: isinstance(x, list))
stack_noise_cl_instr = jax.tree.map(_stack, noise_cl_instr, is_leaf=lambda x: isinstance(x, list))
stack_noise_dl_white = jax.tree.map(_stack, noise_dl_white, is_leaf=lambda x: isinstance(x, list))
stack_noise_dl_instr = jax.tree.map(_stack, noise_dl_instr, is_leaf=lambda x: isinstance(x, list))


# empirical eta (variance increase)
# only load for hwp case since it is the same for the other one
epsilon_data = np.stack(
    [
        np.load(JZ / "opti/var_increase" / f"scatter_{scatter}" / "ml/epsilon_dist.npy")
        for scatter in SCATTERS
    ]
)
etas = (1 / (1 - epsilon_data**2)).mean(axis=-1)


def plot_noise_increase(noise_type: NoiseType, stack_noise_cl_data, relative=True):
    # Determine sharey based on plot type and noise type
    sharey = "row" if not relative and noise_type == NoiseType.INSTR else True

    # Create subplots with determined parameters
    fig, axs = plt.subplots(
        2, 2, figsize=(12, 10), layout="constrained", sharex=True, sharey=sharey
    )

    # Set titles for rows
    axs[0, 0].set_title("EE, HWP on")
    axs[0, 1].set_title("BB, HWP on")
    axs[1, 0].set_title("EE, HWP off")
    axs[1, 1].set_title("BB, HWP off")

    flare = sns.color_palette("flare", as_cmap=True)
    colors = [flare(i / (len(SCATTERS) - 1)) for i in range(len(SCATTERS))]

    # Create empty list to store legend handles
    legend_handles = []

    # Dictionary to store average values
    average_values = {"hwp": {"EE": {}, "BB": {}}, "no_hwp": {"EE": {}, "BB": {}}}

    for row, khwp in enumerate(["hwp", "no_hwp"]):
        cl = stack_noise_cl_data[khwp]

        # Set appropriate axis limits
        for col, idx in enumerate([0, 3]):  # 0 for EE, 3 for BB
            ax = axs[row, col]
            ax.grid(True)

            if col == 0:  # Only set ylabel on first column
                if relative:
                    ax.set_ylabel(r"$N_\ell^\mathsf{pd} / N_\ell^\mathsf{iqu} - 1$")
                    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
                else:
                    ax.set_ylabel(r"$N_\ell^\mathsf{pd} - N_\ell^\mathsf{iqu}$ [$\mu K^2$]")

            if row == 1:  # Only set xlabel on bottom row
                ax.set_xlabel(r"Multipole $\ell$")

            # Plot theory spectra if showing absolute values
            if not relative and col == 1:
                line = ax.plot(
                    ls[ls > 20],
                    prim_BB[ls > 20],
                    "k-",
                    lw=1.0,
                    alpha=0.7,
                    label=r"BB prim ($r=0.01$)",
                )[0]
                # avoid duplicate legend entries
                if row == 0:
                    legend_handles.append(line)

        for i, scatter in enumerate(SCATTERS):
            iqu = cl[scatter]["ml"]
            pd = cl[scatter]["pd"]
            ells = iqu["ells"][0]  # identical for all realizations

            if relative:
                y = pd["cl_22"] / iqu["cl_22"] - 1
            else:
                y = pd["cl_22"] - iqu["cl_22"]

            for col, idx in enumerate([0, 3]):  # 0 for EE, 3 for BB
                spec_type = "EE" if idx == 0 else "BB"
                ax = axs[row, col]
                ymean = y.mean(axis=0)[idx]
                yerr = y.std(axis=0)[idx]

                # Store the average values in the dictionary
                if relative:
                    average_values[khwp][spec_type][scatter] = ymean

                # slightly adjust ells of different scatter values for better visibility
                ells_sep = ells + (i + 0.5 - len(SCATTERS) / 2) * 2
                errorbar = ax.errorbar(
                    ells_sep,
                    ymean,
                    yerr=yerr,
                    fmt=".",
                    color=colors[i],
                    linewidth=0.5,
                    label=f"Scatter {scatter:.1%}",
                )

                # Only add to legend for the first subplot to avoid duplicates
                if row == 0 and col == 0:
                    legend_handles.append(errorbar)

                # Print increase averaged over bins
                if relative:
                    line = ax.axhline(etas[i] - 1, color=colors[i], alpha=0.7, linestyle="--")
                    # Only add to legend for the first subplot to avoid duplicates
                    if row == 0 and col == 0 and i == 0:
                        legend_handles.insert(
                            0,
                            plt.Line2D(
                                [0],
                                [0],
                                linestyle="--",
                                color="gray",
                                label="empirical expectation",
                            ),
                        )

    # Add a figure-level legend at the top
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="outside upper center",
        ncol=len(legend_handles),  # everything on one line
    )

    # Autoscale axes after adding collections
    for ax in axs.flat:
        ax.autoscale()
        ax.set_xlim(0, LMAX)

    # Apply specific ylim settings for absolute plots
    if not relative:
        if noise_type == NoiseType.WHITE:
            axs[0, 0].set_ylim(-1e-7, 1.5e-7)
        else:
            axs[0, 0].set_ylim(-0.2e-7, 2.5e-7)
            axs[1, 0].set_ylim(-1.2e-6, 7e-6)

    # Save the figure
    plot_type = "relative" if relative else "absolute"
    fig.savefig(f"var_increase_spectra_{noise_type.value}_{plot_type}.svg", bbox_inches="tight")
    plt.close(fig)

    # Return the dictionary with average values
    return average_values


# # Plot relative increase with white noise
# plot_noise_increase(NoiseType.WHITE, stack_noise_cl_white, relative=True)

# # Plot absolute increase with white noise
# plot_noise_increase(NoiseType.WHITE, stack_noise_cl_white, relative=False)

# Plot relative increase with instrumental noise
avg_instr = plot_noise_increase(NoiseType.INSTR, stack_noise_cl_instr, relative=True)
np.savetxt(
    "average_increase_bins.txt",
    np.column_stack(
        [
            SCATTERS,
            etas - 1,
            [avg_instr["hwp"]["EE"][scatter].mean() for scatter in SCATTERS],
            [avg_instr["hwp"]["BB"][scatter].mean() for scatter in SCATTERS],
            [avg_instr["no_hwp"]["EE"][scatter].mean() for scatter in SCATTERS],
            [avg_instr["no_hwp"]["BB"][scatter].mean() for scatter in SCATTERS],
        ]
    ),
    delimiter="\t",
    header="scatter\teta-1\tEE hwp\tBB hwp\tEE no_hwp\tBB no_hwp",
    fmt=["%.3f", "%.2e", "%.2e", "%.2e", "%.2e", "%.2e"],
)

# Plot absolute increase with instrumental noise
plot_noise_increase(NoiseType.INSTR, stack_noise_cl_instr, relative=False)


# ___________________________________________________________________

_l = input_dl["ells"]
_fig, _axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
_axs[0].plot(ls[2:], theory_dl["EE"][2:], "k:", label="EE theory")
_axs[0].plot(_l, input_dl["cl_22"][0], label="Input EE")
_axs[1].plot(ls, theory_dl["BB"], "k:", label="BB theory")
_axs[1].plot(_l, input_dl["cl_22"][3], label="Input BB")
_axs[2].plot(_l, input_dl["cl_22"][0] / theory_dl["EE"][_l.astype(int)], label="EE ratio")
_axs[2].plot(_l, input_dl["cl_22"][3] / theory_dl["BB"][_l.astype(int)], label="BB ratio")
_axs[2].set_ylim(0, 1.1)
for _i, _ax in enumerate(_axs):
    _ax.set_xlabel(r"$\ell$")
    if _i < 2:
        _ax.set_ylabel(r"$C_\ell [\mu K^2]$")
    _ax.legend()
_fig.tight_layout()
_fig.savefig("input_vs_theory_spectra.svg", bbox_inches="tight")


# ___________________________________________________________________


def fisher_r0(avg_noise_cl, ells, fsky: float, A_lens: float = 1.0):
    # Get the ells and average noise power spectrum
    N_ell = avg_noise_cl

    # Sample prim_BB and lens_BB at ell values in ell_arr
    prim_BB_sampled = 100 * prim_BB[ells.astype(int)]  # originally for r = 0.01
    lens_BB_sampled = lens_BB[ells.astype(int)]
    ratio = prim_BB_sampled / (A_lens * lens_BB_sampled + N_ell)

    return (0.5 * fsky * np.sum((2 * ells + 1) * ratio**2)) ** -0.5


# Compute increase in Fisher r0 for each scatter value
sigma_r0 = {
    khwp: {
        k_ml_pd: np.array(
            [
                fisher_r0(
                    stack_noise_cl_instr[khwp][scatter][k_ml_pd]["cl_22"][3],
                    stack_noise_cl_instr[khwp][scatter]["ml"]["ells"][0],
                    fsky=0.15,
                )
                for scatter in SCATTERS
            ]
        )
        for k_ml_pd in ["ml", "pd"]
    }
    for khwp in ["hwp", "no_hwp"]
}
np.savetxt(
    "sigma_r0.txt",
    np.column_stack(
        [
            sigma_r0["hwp"]["ml"],
            sigma_r0["hwp"]["pd"],
            sigma_r0["no_hwp"]["ml"],
            sigma_r0["no_hwp"]["pd"],
        ]
    ),
    delimiter="\t",
    header="IQU hwp\tPD hwp\tIQU no hwp\tPD no hwp",
    fmt="%.6e",
)
