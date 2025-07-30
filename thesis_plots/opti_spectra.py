#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
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


cartview = partial(
    hp.cartview, unit=r"$\mu K$", xsize=5000, latra=[-70, -10], lonra=[-30, 90], cmap="bwr"
)


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

# ells_full = input_dl["ells"].astype(int)
# fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
# axs[0].plot(ls[2:], theory_dl["EE"][2:], "k:", label="EE theory")
# axs[0].plot(ells_full, input_dl["cl_22"][0], label="Input EE")
# axs[1].plot(ls, theory_dl["BB"], "k:", label="BB theory")
# axs[1].plot(ells_full, input_dl["cl_22"][3], label="Input BB")
# axs[2].plot(ells_full, input_dl["cl_22"][0] / theory_dl["EE"][ells_full], label="EE ratio")
# axs[2].plot(ells_full, input_dl["cl_22"][3] / theory_dl["BB"][ells_full], label="BB ratio")
# axs[2].set_ylim(0, 1.1)
# for i, ax in enumerate(axs):
#     ax.set_xlabel(r"$\ell$")
#     if i < 2:
#         ax.set_ylabel(r"$C_\ell [\mu K^2]$")
#     ax.legend()
#     ax.grid(True)
# fig.tight_layout()
# fig.savefig("input_vs_theory_spectra.svg", bbox_inches="tight")


# ___________________________________________________________________
# Synthetic Q/U maps from pure B-mode simulation

# zeros = np.zeros_like(prim_BB)
# almTEB = hp.synalm([zeros, zeros, prim_BB, zeros], new=True)
# mapIQU = hp.alm2map(almTEB, 512, pol=True)

# fig = plt.figure(figsize=(10, 5))
# cartview(mapIQU[1], title="Q", sub=121, fig=fig)
# cartview(mapIQU[2], title="U", sub=122, fig=fig)
# fig.savefig("pure_B_r001.svg", dpi=150, bbox_inches="tight")


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
    # Create subplots with determined parameters
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout="constrained", sharex=True, sharey="row")

    # Set titles for rows
    axs[0, 0].set_title("EE, HWP on")
    axs[0, 1].set_title("BB, HWP on")
    axs[1, 0].set_title("EE, HWP off")
    axs[1, 1].set_title("BB, HWP off")

    if relative and noise_type == "instr":
        inset_axs = [ax.inset_axes([0.45, 0.4, 0.5, 0.4]) for ax in axs[1, :]]
        for ax, inset in zip(axs[1, :], inset_axs):
            # Draw box around the region that's being zoomed
            ax.indicate_inset_zoom(inset, edgecolor="black")
            inset.grid(True, alpha=0.5)
            inset.set_xlim(200, 400)
            inset.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

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
                    ax.set_ylabel(
                        r"$(N_\ell^\mathsf{pd} - N_\ell^\mathsf{iqu}) / \langle N_\ell^\mathsf{iqu} \rangle$"
                    )
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

                # estimate white noise level as average of the last 10% of ell bins
                white_noise_level = np.mean(iqu[-len(ells) // 10 :])
                # print(
                #     f"White noise level for {khwp} {('EE', 'BB')[col]} {scatter:.1%}: {white_noise_level:.5e}"
                # )

                diff = pd - iqu
                if relative:
                    diff /= white_noise_level

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

                if row == 1 and relative and noise_type == "instr":
                    inset = inset_axs[col]
                    inset.errorbar(
                        ells_sep[(ells_sep > 200) & (ells_sep < 400)],
                        ymean[(ells_sep > 200) & (ells_sep < 400)],
                        yerr=yerr[(ells_sep > 200) & (ells_sep < 400)],
                        fmt=".",
                        color=colors[i],
                        linewidth=0.5,
                    )

                # Print increase averaged over bins
                # if relative:
                #     line = ax.axhline(etas[i] - 1, color=colors[i], alpha=0.7, linestyle="--")
                #     # Only add to legend for the first subplot to avoid duplicates
                #     if row == 0 and col == 0 and i == 0:
                #         legend_handles.insert(
                #             0,
                #             plt.Line2D(  # pyright: ignore[reportPrivateImportUsage]
                #                 [0],
                #                 [0],
                #                 linestyle="--",
                #                 color="gray",
                #                 label="empirical expectation",
                #             ),
                #         )

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
# plot_noise_increase("white", relative=True)

# Plot absolute increase with white noise
# plot_noise_increase("white", relative=False)

# Plot relative increase with instrumental noise
plot_noise_increase("instr", relative=True)

# Plot absolute increase with instrumental noise
# plot_noise_increase("instr", relative=False)


# ___________________________________________________________________
# Inspect low-ell bins in no HWP case and scatter of 1%

scatter_idx = SCATTERS.index(0.01)  # 1% scatter
khwp = "no_hwp"

# Get the data
ells = noise_cl["instr"][khwp]["ml"][0.01]["ells"][0]
iqu_ee = noise_cl["instr"][khwp]["ml"][0.01]["cl_22"][:, 0]  # EE
pd_ee = noise_cl["instr"][khwp]["pd"][0.01]["cl_22"][:, 0]  # EE
iqu_bb = noise_cl["instr"][khwp]["ml"][0.01]["cl_22"][:, 3]  # BB
pd_bb = noise_cl["instr"][khwp]["pd"][0.01]["cl_22"][:, 3]  # BB

white_noise_level_ee = np.mean(iqu_ee[-len(ells) // 10 :])
white_noise_level_bb = np.mean(iqu_bb[-len(ells) // 10 :])

# Calculate differences (PD - IQU) with respect to white noise level
diff_ee = (pd_ee - iqu_ee) / white_noise_level_ee
diff_bb = (pd_bb - iqu_bb) / white_noise_level_bb

# Create figure with 2 subplots for the differences in the first ell bin
fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharex=True, sharey=True)

# Get values for the first ell bin for both spectra types
first_bin_idx = 0  # First ell bin
diff_ee_first_bin = diff_ee[:, first_bin_idx]
diff_bb_first_bin = diff_bb[:, first_bin_idx]

# Plot EE differences
axs[0].set_title("EE")
for i in range(len(diff_ee_first_bin)):
    axs[0].scatter([i], [diff_ee_first_bin[i]], color="purple", marker="o")

# Add mean line and shaded region for standard deviation
diff_ee_mean = diff_ee_first_bin.mean()
diff_ee_std = diff_ee_first_bin.std()

x_range = np.array([-1, len(diff_ee_first_bin)])
axs[0].axhline(diff_ee_mean, color="purple", linestyle="dashed", linewidth=2, label="Mean")
axs[0].axhline(0, color="black", linestyle="solid", linewidth=1, alpha=0.5, label="Zero line")
axs[0].fill_between(
    x_range,
    [diff_ee_mean - diff_ee_std],
    [diff_ee_mean + diff_ee_std],
    color="purple",
    alpha=0.2,
    label="±1σ",
)

axs[0].set_xlabel("Realization index")
axs[0].set_ylabel(r"$(N_\ell^\mathsf{pd} - N_\ell^\mathsf{iqu}) / \langle N_\ell^\mathsf{iqu} \rangle$")
axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
axs[0].set_xlim(-1, len(diff_ee_first_bin))
axs[0].grid(True)

# Plot BB differences
axs[1].set_title("BB")
for i in range(len(diff_bb_first_bin)):
    axs[1].scatter([i], [diff_bb_first_bin[i]], color="purple", marker="o")

# Add mean line and shaded region for standard deviation
diff_bb_mean = diff_bb_first_bin.mean()
diff_bb_std = diff_bb_first_bin.std()

axs[1].axhline(diff_bb_mean, color="purple", linestyle="dashed", linewidth=2, label="Mean")
axs[1].axhline(0, color="black", linestyle="solid", linewidth=1, alpha=0.5, label="Zero line")
axs[1].fill_between(
    x_range,
    [diff_bb_mean - diff_bb_std],
    [diff_bb_mean + diff_bb_std],
    color="purple",
    alpha=0.2,
    label="±1σ",
)

axs[1].set_xlabel("Realization index")
axs[1].set_xlim(-1, len(diff_bb_first_bin))
axs[1].grid(True)

# Create figure-level legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3)

# Add text with statistics
for i, (diff_data, iqu_data, title) in enumerate(
    [
        (diff_ee_first_bin, iqu_ee[:, first_bin_idx], "EE"),
        (diff_bb_first_bin, iqu_bb[:, first_bin_idx], "BB"),
    ]
):
    stats_text = (
        f"Mean: {diff_data.mean():.2%}\nStddev: {diff_data.std():.2%}\n"
        # f"Relative diff: {(diff_data.mean() / iqu_data.mean()):.2%}"
    )
    axs[i].text(
        0.05,
        0.95,
        stats_text,
        transform=axs[i].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

fig.savefig("noise_difference_first_bin_no_hwp_1pct.svg", bbox_inches="tight", dpi=150)
plt.close(fig)

# ___________________________________________________________________
# Increased uncertainty on r ?


def fisher_r0(N_ell, ell_binned, fsky: float, A_lens: float = 1.0):
    # Sample prim_BB and lens_BB at ell values in ell_arr
    prim_BB_sampled = 100 * prim_BB[ell_binned]  # originally for r = 0.01
    lens_BB_sampled = lens_BB[ell_binned]
    ratio = prim_BB_sampled / (A_lens * lens_BB_sampled + N_ell)

    # compute for each realization (so do not sum over axis 0)
    return (0.5 * fsky * np.sum((2 * ell_binned + 1) * ratio**2, axis=-1)) ** -0.5


# Compute increase in Fisher r0 for each scatter value
# wl.pprint(noise_cl["instr"]["hwp"])
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
# wl.pprint(sigma_r0)
np.savetxt(
    "sigma_r0_avg.csv",
    np.column_stack(
        [
            np.array(SCATTERS) * 100,
            hwp_ml_avg := sigma_r0["hwp"]["ml"].mean(axis=1),
            hwp_pd_avg := sigma_r0["hwp"]["pd"].mean(axis=1),
            (hwp_pd_avg / hwp_ml_avg - 1) * 100,
            no_hwp_ml_avg := sigma_r0["no_hwp"]["ml"].mean(axis=1),
            no_hwp_pd_avg := sigma_r0["no_hwp"]["pd"].mean(axis=1),
            (no_hwp_pd_avg / no_hwp_ml_avg - 1) * 100,
        ]
    ),
    delimiter=",",
    header="Scatter (%),IQU hwp,PD hwp,rel increase HWP (%),IQU no hwp,PD no hwp,rel increase no HWP (%)",
    fmt=("%.1f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f"),
)

# Save a second file with standard deviations
np.savetxt(
    "sigma_r0_std.csv",
    np.column_stack(
        [
            np.array(SCATTERS) * 100,
            sigma_r0["hwp"]["ml"].std(axis=1),
            sigma_r0["hwp"]["pd"].std(axis=1),
            sigma_r0["no_hwp"]["ml"].std(axis=1),
            sigma_r0["no_hwp"]["pd"].std(axis=1),
        ]
    ),
    delimiter=",",
    header="Scatter (%),IQU hwp std,PD hwp std,IQU no hwp std,PD no hwp std",
    fmt=("%.1f", "%.18f", "%.18f", "%.18f", "%.18f"),
)
