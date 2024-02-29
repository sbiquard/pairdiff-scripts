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


def process(dirname):
    # Create 'plots' directory if needed
    run = pathlib.Path(dirname)
    plotdir = run / "plots"
    plotdir.mkdir(exist_ok=True)

    # Read spectra
    cldir = run / "spectra"
    full_cl = np.load(cldir / "full_cl.npz")
    noise_cl = np.load(cldir / "noise_cl.npz")
    ell_arr = full_cl["ell_arr"]

    def plot(cl, ax, ls):
        cl_00 = cl.get("cl_00")
        cl_02 = cl.get("cl_02")
        cl_22 = cl.get("cl_22")
        if cl_00 is not None:
            ax.plot(ell_arr, cl_00[0], "r", linestyle=ls)
        if cl_02 is not None:
            ax.plot(ell_arr, np.fabs(cl_02[0]), "y", linestyle=ls)
        ax.plot(ell_arr, cl_22[0], "g", linestyle=ls)
        ax.plot(ell_arr, cl_22[3], "b", linestyle=ls)

    fig, ax = plt.subplots()
    fig.suptitle("Power spectra and noise spectra of estimated maps")
    plot(full_cl, ax, "solid")
    plot(noise_cl, ax, "dashed")
    ax.loglog()
    ax.set_xlim(left=20)
    ax.set_xlabel("$\\ell$", fontsize=16)
    ax.set_ylabel("$C_\\ell$", fontsize=16)
    ax.legend(
        CUSTOM_LINES,
        ["TT", "TE", "EE", "BB", "full", "noise"],
        loc="upper right",
        ncol=2,
        labelspacing=0.1,
    )
    plt.savefig(plotdir / "spectra.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save power spectra for one run.",
    )
    parser.add_argument("dirname", type=str, help="name of directory")
    args = parser.parse_args()
    process(args.dirname)
