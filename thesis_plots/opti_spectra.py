#!/usr/bin/env python3

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

# import wadler_lindig as wl
from theory_spectra import get_theory_powers

sns.set_theme(context="notebook", style="ticks")


JZ = Path(__file__).parents[1].absolute() / "jz_out"
OPTI = JZ / "opti"
SCATTERS = [0.001, 0.01, 0.1, 0.2]
NUM_REAL = 25
HITS = 10_000
LMIN, LMAX = 30, 500


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    # remove unnecessary dimensions
    return {k: data[k].squeeze() for k in data}


def cl2dl(ell, cl):
    return ell * (ell + 1) / 2 / np.pi * cl


def load_spectra(path: Path, only_22: bool = True, convert_to_dl: bool = False):
    cl = load_npz(path)
    ell = cl.pop("ell_arr").astype(np.int32)
    if convert_to_dl:
        cl = jax.tree.map(lambda x: cl2dl(ell, x), cl)
    good = (LMIN <= ell) & (ell <= LMAX)
    if only_22:
        return {"ells": ell[good], "cl_22": cl["cl_22"][..., good]}
    return {"ells": ell[good], **{k: cl[k][..., good] for k in cl}}


def stack_dict(cl: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    # turn list of dicts into a single dict with stacked arrays
    return {k: np.stack(tuple(cl[i][k] for i in range(len(cl)))) for k in cl[0]}


# ___________________________________________________________________
# Theory spectra

theory_cl = get_theory_powers(lmax=LMAX)
ls = np.arange(theory_cl["BB"].size)
theory_dl = jax.tree.map(lambda x: cl2dl(ls, x), theory_cl)
lens_BB = get_theory_powers(r=0, lmax=LMAX)["BB"]
prim_BB = theory_cl["BB"] - lens_BB
prim_BB_dl = cl2dl(ls, prim_BB)

input_cl = load_spectra(JZ / "input_cells_mask_apo_1000.npz")
input_dl = load_spectra(JZ / "input_cells_mask_apo_1000.npz", convert_to_dl=True)

ells_full = input_dl["ells"].astype(int)
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
axs[0].plot(ls[2:], theory_dl["EE"][2:], "k:", label="EE theory")
axs[0].plot(ells_full, input_dl["cl_22"][0], label="Input EE")
axs[1].plot(ls, theory_dl["BB"], "k:", label="BB theory")
axs[1].plot(ells_full, input_dl["cl_22"][3], label="Input BB")
axs[2].plot(ells_full, input_dl["cl_22"][0] / theory_dl["EE"][ells_full], label="EE ratio")
axs[2].plot(ells_full, input_dl["cl_22"][3] / theory_dl["BB"][ells_full], label="BB ratio")
axs[2].set_ylim(0, 1.1)
for i, ax in enumerate(axs):
    ax.set_xlabel(r"$\ell$")
    if i < 2:
        ax.set_ylabel(r"$C_\ell [\mu K^2]$")
    ax.legend()
    ax.grid(True)
fig.tight_layout()
fig.savefig("input_vs_theory_spectra.svg", bbox_inches="tight")


# ___________________________________________________________________
# Optimality runs


def get_run_path(k_ml_pd: str, k_noise: str, k_hwp: str, scatter: float, real: int) -> Path:
    noise_suffix = "" if k_noise == "white" else "_instr"
    hwp_suffix = "" if k_hwp == "hwp" else "_no_hwp"
    folder = OPTI / ("var_increase" + noise_suffix + hwp_suffix)
    return folder / f"{real + 1:03d}" / f"scatter_{scatter}" / k_ml_pd


noise_cl = {
    k_noise: {
        k_hwp: {
            k_ml_pd: {
                scatter: stack_dict(
                    [
                        load_spectra(
                            get_run_path(k_ml_pd, k_noise, k_hwp, scatter, real)
                            / "spectra"
                            / f"noise_cl_hits_{HITS}.npz"
                        )
                        for real in range(NUM_REAL)
                    ]
                )
                for scatter in SCATTERS
            }
            for k_ml_pd in ["ml", "pd"]
        }
        for k_hwp in ["hwp", "no_hwp"]
    }
    for k_noise in ["white", "instr"]
}
# wl.pprint(noise_cl)

# empirical eta (variance increase)
# only load for hwp case since it is the same for the other one
epsilon_data = np.stack(
    [
        np.load(JZ / "opti/var_increase" / f"scatter_{scatter}" / "ml/epsilon_dist.npy")
        for scatter in SCATTERS
    ]
)
etas = (1 / (1 - epsilon_data**2)).mean(axis=-1)


def plot_noise_increase(noise_type: str, relative: bool = True):
    # Determine sharey based on plot type and noise type
    sharey = "row" if not relative and noise_type == "instr" else True

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

    for row, khwp in enumerate(["hwp", "no_hwp"]):
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
                # ells don't change from realization to realization
                ells = noise_cl[noise_type][khwp]["ml"][scatter]["ells"][0]
                iqu = noise_cl[noise_type][khwp]["ml"][scatter]["cl_22"][:, idx]
                pd = noise_cl[noise_type][khwp]["pd"][scatter]["cl_22"][:, idx]

                diff = pd - iqu
                if relative:
                    diff /= iqu

                ymean = diff.mean(axis=0)
                yerr = diff.std(axis=0)

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
                            plt.Line2D(  # pyright: ignore[reportPrivateImportUsage]
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
        if noise_type == "white":
            axs[0, 0].set_ylim(-1e-7, 1.5e-7)
        else:
            axs[0, 0].set_ylim(-0.2e-7, 2.5e-7)
            axs[1, 0].set_ylim(-1.2e-6, 7e-6)

    # Save the figure
    plot_type = "relative" if relative else "absolute"
    fig.savefig(f"var_increase_spectra_{noise_type}_{plot_type}.svg", bbox_inches="tight")
    plt.close(fig)


# Plot relative increase with white noise
plot_noise_increase("white", relative=True)

# Plot absolute increase with white noise
plot_noise_increase("white", relative=False)

# Plot relative increase with instrumental noise
plot_noise_increase("instr", relative=True)

# Plot absolute increase with instrumental noise
plot_noise_increase("instr", relative=False)


# ___________________________________________________________________
# Average increase across bins


def f(noise_type: str, k_hwp: str, spec_type: str, scatter: float):
    idx = 0 if spec_type == "EE" else 3
    iqu = noise_cl[noise_type][k_hwp]["ml"][scatter]["cl_22"][:, idx]
    pd = noise_cl[noise_type][k_hwp]["pd"][scatter]["cl_22"][:, idx]
    rel_diff = pd / iqu - 1
    return rel_diff.mean()


average_relative_increase = {
    k_noise: {
        k_hwp: {
            spec_type: [f(k_noise, k_hwp, spec_type, scatter) for scatter in SCATTERS]
            for spec_type in ["EE", "BB"]
        }
        for k_hwp in ["hwp", "no_hwp"]
    }
    for k_noise in ["white", "instr"]
}

for k, v in average_relative_increase.items():
    np.savetxt(
        f"average_increase_bins_{k}.csv",
        np.column_stack(
            [
                SCATTERS,
                etas - 1,
                v["hwp"]["EE"],
                v["hwp"]["BB"],
                v["no_hwp"]["EE"],
                v["no_hwp"]["BB"],
            ]
        ),
        delimiter=",",
        header="scatter,eta-1,EE hwp,BB hwp,EE no_hwp,BB no_hwp",
        fmt=["%.3f", "%.2e", "%.2e", "%.2e", "%.2e", "%.2e"],
    )


# ___________________________________________________________________
# Increased uncertainty on r ?


def fisher_r0(N_ell, ell_binned, fsky: float, A_lens: float = 1.0):
    # Sample prim_BB and lens_BB at ell values in ell_arr
    prim_BB_sampled = 100 * prim_BB[ell_binned]  # originally for r = 0.01
    lens_BB_sampled = lens_BB[ell_binned]
    ratio = prim_BB_sampled / (A_lens * lens_BB_sampled + N_ell)

    return (0.5 * fsky * np.sum((2 * ell_binned + 1) * ratio**2)) ** -0.5


# Compute increase in Fisher r0 for each scatter value
sigma_r0 = {
    khwp: {
        k_ml_pd: np.array(
            [
                fisher_r0(
                    noise_cl["instr"][khwp][k_ml_pd][scatter]["cl_22"][:, 3],
                    noise_cl["instr"][khwp]["ml"][scatter]["ells"][0],
                    fsky=0.15,
                )
                for scatter in SCATTERS
            ]
        )
        for k_ml_pd in ["ml", "pd"]
    }
    for khwp in ["hwp", "no_hwp"]
}
# wl.pprint(sigma_r0, short_arrays=False)
np.savetxt(
    "sigma_r0.csv",
    np.column_stack(
        [
            np.array(SCATTERS) * 100,
            hwp_ml := sigma_r0["hwp"]["ml"],
            hwp_pd := sigma_r0["hwp"]["pd"],
            (hwp_pd / hwp_ml - 1) * 100,
            no_hwp_ml := sigma_r0["no_hwp"]["ml"],
            no_hwp_pd := sigma_r0["no_hwp"]["pd"],
            (no_hwp_pd / no_hwp_ml - 1) * 100,
        ]
    ),
    delimiter=",",
    header="Scatter (%),IQU hwp,PD hwp,rel increase HWP (%),IQU no hwp,PD no hwp,rel increase no HWP (%)",
    fmt=("%.1f","%.18f","%.18f","%.18f","%.18f","%.18f","%.18f"),
)
