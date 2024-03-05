#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CUSTOM_LINES = [
    Line2D([0], [0], color="r"),
    Line2D([0], [0], color="y"),
    Line2D([0], [0], color="g"),
    Line2D([0], [0], color="b"),
    Line2D([0], [0], color="k", linestyle="solid"),
    Line2D([0], [0], color="k", linestyle="dashed"),
]

CUSTOM_LABELS = ["TT", "TE", "EE", "BB", "full", "noise"]


def plot(cl, ax, ls="solid", dl=False):
    ell_arr = cl["ell_arr"]
    cl_00 = cl.get("cl_00")
    cl_02 = cl.get("cl_02")
    cl_22 = cl.get("cl_22")

    def _plot(temp, cross, pol, ells):
        has_T = False
        if temp is not None:
            has_T = True
            ax.plot(ells, temp[0], "r", linestyle=ls)
        if cross is not None:
            ax.plot(ells, np.fabs(cross[0]), "y", linestyle=ls)
        ax.plot(ells, pol[0], "g", linestyle=ls)
        ax.plot(ells, pol[3], "b", linestyle=ls)
        return has_T

    has_T: bool
    if dl:
        has_T = _plot(
            cl_00 * ell_arr * (ell_arr + 1) / (2 * np.pi),
            cl_02 * ell_arr * (ell_arr + 1) / (2 * np.pi),
            cl_22 * ell_arr * (ell_arr + 1) / (2 * np.pi),
            ell_arr,
        )
    else:
        has_T = _plot(cl_00, cl_02, cl_22, ell_arr)
    return has_T


def decorate(ax, has_T: bool = False, dl: bool = False):
    ax.loglog()
    ax.set_xlim(left=20)
    ax.set_xlabel("$\\ell$", fontsize=16)
    ylabel: str
    if dl:
        ylabel = "$\\ell(\\ell+1) \\times C_\\ell / 2\\pi$"
    else:
        ylabel = "$C_\\ell$"
    ax.set_ylabel(ylabel, fontsize=16)
    slc = slice(None if has_T else 2, None)
    ax.legend(
        CUSTOM_LINES[slc],
        CUSTOM_LABELS[slc],
        loc="upper right",
        ncol=2,
        labelspacing=0.1,
    )


def process(dirname, ref: str = ""):
    # Create 'plots' directory if needed
    run = pathlib.Path(dirname)
    plotdir = run / "plots"
    plotdir.mkdir(exist_ok=True)

    # Read spectra
    cldir = run / "spectra"
    full_cl = np.load(cldir / f"full_cl{ref}.npz")
    noise_cl = np.load(cldir / f"noise_cl{ref}.npz")

    fig, ax = plt.subplots()
    fig.suptitle("Power spectra and noise spectra of estimated maps")
    has_tt = plot(full_cl, ax, ls="solid")
    plot(noise_cl, ax, ls="dashed")
    decorate(ax, has_tt)
    plt.savefig(plotdir / "spectra.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save power spectra for one run.",
    )
    parser.add_argument("dirname", type=str, help="name of directory")
    args = parser.parse_args()
    process(args.dirname)
