#!/usr/bin/env python3

from enum import Enum
from pathlib import Path

import jax
import matplotlib as mpl
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
# SCATTERS = [0.001, 0.01]
# MASK_REF = "hits_10000"
MASK_REF = "hits_1000"


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
                    for real in range(25)
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


def load_spectra(path: Path | list[Path], lmax=1000, dl=False):
    cl = load_npz(path)
    ell = cl.pop("ell_arr")
    good = ell <= lmax
    if dl:
        return {"ells": ell[good], **{k: cl2dl(ell[good], cl[k][..., good]) for k in cl}}
    else:
        return {"ells": ell[good], **{k: cl[k][..., good] for k in cl}}


theory_cl = get_theory_powers(r=1.0)
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
    # assuming homogeneous shapes
    return {k: np.stack(tuple(cl[i][k] for i in range(len(cl)))) for k in cl[0]}


stack_full_cl_white = jax.tree.map(_stack, full_cl_white, is_leaf=lambda x: isinstance(x, list))
stack_full_cl_instr = jax.tree.map(_stack, full_cl_instr, is_leaf=lambda x: isinstance(x, list))
stack_full_dl_white = jax.tree.map(_stack, full_dl_white, is_leaf=lambda x: isinstance(x, list))
stack_full_dl_instr = jax.tree.map(_stack, full_dl_instr, is_leaf=lambda x: isinstance(x, list))
stack_noise_cl_white = jax.tree.map(_stack, noise_cl_white, is_leaf=lambda x: isinstance(x, list))
stack_noise_cl_instr = jax.tree.map(_stack, noise_cl_instr, is_leaf=lambda x: isinstance(x, list))
stack_noise_dl_white = jax.tree.map(_stack, noise_dl_white, is_leaf=lambda x: isinstance(x, list))
stack_noise_dl_instr = jax.tree.map(_stack, noise_dl_instr, is_leaf=lambda x: isinstance(x, list))


def plot_noise_increase(noise_type: NoiseType, stack_noise_cl_data, relative=True, r=0.01):
    fig = plt.figure(layout="constrained", figsize=(12, 10))
    # fig.suptitle(
    #     "{} noise power increase in pair differencing maps with {} noise".format(
    #         "Relative" if relative else "Absolute", noise_type
    #     )
    # )

    subfigs = fig.subfigures(2, 1)
    subfigs[0].suptitle("HWP on")
    subfigs[1].suptitle("HWP off")

    cmap = sns.color_palette("flare", as_cmap=True)
    colors = [cmap(i / (len(SCATTERS) - 1)) for i in range(len(SCATTERS))]

    for khwp, subfig in zip(["hwp", "no_hwp"], subfigs):
        cl = stack_noise_cl_data[khwp]
        axs = subfig.subplots(1, 2, sharex=True, sharey=True)
        axs[0].set_title("EE")
        axs[1].set_title("BB")

        # Set appropriate axis limits
        for ax in axs:
            ax.set_xlim(2, 1000)
            ax.grid(True)
            ax.set_xlabel(r"Multipole $\ell$")
            if relative:
                ax.set_ylabel("Relative power increase")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            else:
                ax.set_ylabel(r"Power $[\mu K^2]$")
            ax.label_outer()

            # Plot theory spectra if showing absolute values
            if not relative:
                # Only plot theory BB spectrum on the second axis
                # Only plot for ell > 20
                mask = ls > 20
                axs[1].plot(
                    ls[mask], r * theory_cl["BB"][mask], "k-", lw=1.0, alpha=0.7, label="Theory BB"
                )
                axs[1].legend()

        for is_, scatter in enumerate(SCATTERS):
            iqu = cl[scatter]["ml"]
            pd = cl[scatter]["pd"]
            ells = iqu["ells"][0]  # identical for all realizations
            good = ells > 30

            if relative:
                diff = pd["cl_22"] / iqu["cl_22"] - 1
            else:
                diff = pd["cl_22"] - iqu["cl_22"]

            for ax, idx in zip(axs, (0, 3)):
                y = diff.mean(axis=0)[idx]
                yerr = diff.std(axis=0)[idx]
                ax.errorbar(
                    ells[good] + (is_ + 0.5 - len(SCATTERS) / 2) * 2,
                    y[good],
                    yerr=yerr[good],
                    fmt=".",
                    color=colors[is_],
                    linewidth=0.5,
                )

                # # Print increase averaged over bins
                # if relative:
                #     print(khwp, scatter, "EE" if idx == 0 else "BB", f"{y.mean():.2%}")

        # Add colorbar with scatter values
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=mpl.colors.Normalize(vmin=min(SCATTERS), vmax=max(SCATTERS), clip=True)
        )
        cbar = fig.colorbar(sm, ax=axs, pad=0.01)
        cbar.set_ticks(SCATTERS)
        cbar.set_ticklabels([f"{s:.1%}" for s in SCATTERS])
        cbar.set_label("Scatter")

        # Autoscale axes after adding collections
        axs[0].autoscale()
        axs[1].autoscale()

        # Apply specific ylim settings for certain plots
        if noise_type == NoiseType.WHITE and relative:
            if khwp == "hwp":
                axs[1].set_ylim(top=0.2)
            else:
                axs[1].set_ylim(-0.1, 0.4)
        elif noise_type == NoiseType.INSTR and not relative and khwp == "no_hwp":
            axs[1].set_ylim(-5e-7, 8e-6)

        for ax in axs:
            ax.set_xlim(right=600)

    # Save the figure
    plot_type = "relative" if relative else "absolute"
    fig.savefig(f"var_increase_spectra_{noise_type.value}_{plot_type}.svg", bbox_inches="tight")
    plt.close(fig)
    # plt.show()


# Plot relative increase with white noise
plot_noise_increase(NoiseType.WHITE, stack_noise_cl_white, relative=True)

# Plot absolute increase with white noise
plot_noise_increase(NoiseType.WHITE, stack_noise_cl_white, relative=False)

# Plot relative increase with instrumental noise
plot_noise_increase(NoiseType.INSTR, stack_noise_cl_instr, relative=True)

# Plot absolute increase with instrumental noise
plot_noise_increase(NoiseType.INSTR, stack_noise_cl_instr, relative=False)

# ___________________________________________________________________
# from utils import read_input_sky, read_mask

# sky = read_input_sky()
# mask_apo = read_mask(JZ / "mask_apo_1000.fits")

# plt.figure(figsize=(18, 8))
# hp.mollview(np.where(mask_apo > 0, sky[0] * mask_apo, np.nan), sub=131, title="input I", cmap="bwr")
# hp.mollview(np.where(mask_apo > 0, sky[1] * mask_apo, np.nan), sub=132, title="input Q", cmap="bwr")
# hp.mollview(np.where(mask_apo > 0, sky[2] * mask_apo, np.nan), sub=133, title="input U", cmap="bwr")
# plt.gcf()


# ___________________________________________________________________
# _l = input_dl["ells"]
# _fig, _axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
# _axs[0].plot(ls[2:], theory_dl["EE"][2:], "k:", label="EE theory")
# _axs[0].plot(_l, input_dl["cl_22"][0], label="Input EE")
# _axs[1].plot(ls, theory_dl["BB"], "k:", label="BB theory")
# _axs[1].plot(_l, input_dl["cl_22"][3], label="Input BB")
# _axs[2].plot(_l, input_dl["cl_22"][0] / theory_dl["EE"][_l.astype(int)], label="EE ratio")
# _axs[2].plot(_l, input_dl["cl_22"][3] / theory_dl["BB"][_l.astype(int)], label="BB ratio")
# _axs[2].set_ylim(0, 1.1)
# for _i, _ax in enumerate(_axs):
#     _ax.set_xlabel(r"$\ell$")
#     if _i < 2:
#         _ax.set_ylabel(r"$C_\ell [\mu K^2]$")
#     _ax.legend()
# _fig.tight_layout()
# plt.show()


def fisher(ell_arr, N_ell, fsky: float, lmin: int = 25, A_lens: float = 1.0):
    # TODO: bin the theory spectra to the same ell bins as the measured spectra
    ratio = prim_BB / (A_lens * lens_BB + N_ell)
    # TODO: factor in the sum should be summed over the bin?
    return (0.5 * fsky * np.sum((2 * ell_arr + 1) * ratio**2)) ** -0.5
