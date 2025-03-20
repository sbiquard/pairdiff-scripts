#!/usr/bin/env python3

import argparse
import functools
import time
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import utils
from astropy.visualization import hist

LONRA = [-95, 135]
LATRA = [-70, -10]

cartview = functools.partial(hp.cartview, lonra=LONRA, latra=LATRA)


def plot_hits_cond(hits, cond, savedir, cmap="bwr"):
    def plot(m, title):
        cartview(m, norm="hist", title=title, cmap=cmap)

    # Plot hits
    plot(hits, "Hits map")
    plt.savefig(savedir / "hits.png")
    plt.close()

    # Plot cond
    plot(cond, "Inverse condition number map")
    plt.savefig(savedir / "cond.png")
    plt.close()


def plot_res_hist(maps, sky_in, savedir):
    resid = {}
    convert = {"I": 0, "Q": 1, "U": 2}

    # Compute the residuals
    for k, v in maps.items():
        resid[k] = sky_in[convert[k]] - v

    def plot_hist_stoke(stokes: str):
        fig, ax = plt.subplots(figsize=(7, 7))

        for stoke in stokes:
            residual = resid[stoke]
            residual = residual[~np.isnan(residual)]
            hist(
                residual,
                bins="scott",  # pyright: ignore[reportArgumentType]
                label=f"{stoke} ; {np.mean(residual):.2e} +/- {np.std(residual):.2e} $\\mu K$",
                histtype="step",
                ax=ax,
            )

        fig.suptitle(f"Histograms of {stokes!r} residuals")
        ax.set_xlabel("$\\mu K_{CMB}$")
        ax.grid(True)
        ax.legend()
        fig.savefig(savedir / f"diff_histograms_{stokes}.png")
        plt.close(fig)

    # plot I separately
    if "I" in maps:
        plot_hist_stoke("I")
    plot_hist_stoke("QU")


def plot_maps(
    maps,
    sky_in,
    savedir: Path,
    cmap: str = "bwr",
    map_range_T: int = 500,
    map_range_P: int = 10,
    diff_range_T: int | None = None,
    diff_range_P: int | None = None,
):
    nrow, ncol = 3, 3
    fig = plt.figure(figsize=(8 * ncol, 4 * nrow))

    unit = "$\\mu K_{CMB}$"
    convert = {"I": 0, "Q": 1, "U": 2}

    for i, stokes in enumerate(convert):
        map_range = map_range_T if i == 0 else map_range_P
        diff_range = diff_range_T if i == 0 else diff_range_P

        # Plot input sky
        cartview(
            sky_in[i],
            title=f"Input {stokes}",
            sub=[nrow, ncol, 1 + 3 * i],
            notext=False,
            min=-map_range,
            max=map_range,
            cmap=cmap,
            unit=unit,
        )

        if stokes in maps:
            # Plot reconstructed map
            cartview(
                maps[stokes],
                title=f"Reconstructed {stokes} map",
                sub=[nrow, ncol, 1 + 3 * i + 1],
                notext=False,
                min=-map_range,
                max=map_range,
                cmap=cmap,
                unit=unit,
            )

            # Plot difference map
            diff = sky_in[i] - maps[stokes]
            offset = np.nanmedian(diff)
            rms = np.nanstd(diff)
            amp = 2 * rms
            cartview(
                diff,
                title=f"Difference {stokes}",
                sub=[nrow, ncol, 1 + 3 * i + 2],
                notext=False,
                cmap="bwr",
                min=-(diff_range or -(offset - amp)),
                max=diff_range or (offset + amp),
                unit=unit,
            )

    fig.savefig(savedir / "maps.png")
    plt.close(fig)


def plot_residuals(data, savedir):
    fig, ax = plt.subplots(figsize=[8, 6])
    if data.size < 30:
        fmt = "ko--"
    else:
        fmt = "k--"
    ax.semilogy(np.arange(len(data)), np.sqrt(data / data[0]), fmt, label="PCG")
    ax.set_title("PCG residuals evolution")
    ax.set_xlabel("Step")
    ax.set_ylabel("||r|| / ||r0||")
    ax.grid(True)
    ax.legend()
    fig.savefig(savedir / "pcg_residuals.png")
    plt.close(fig)


def process(args):
    # start timer
    tic = time.perf_counter()

    # create folder for the plots
    run = Path(args.dirname)
    plotdir = run / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    ref = utils.get_last_ref(run)

    # read data
    maps = utils.read_maps(args.dirname, ref=ref)
    hits, cond = utils.read_hits_cond(args.dirname, ref=ref)
    residuals = utils.read_residuals(args.dirname, ref=ref)
    sky_in = utils.read_input_sky()

    # define a mask for pixels outside the solved patch
    thresh = np.percentile(hits[hits > 0], args.hits_percentile)
    mask = hits < thresh
    for m in maps.values():
        m[mask] = np.nan
    cond[mask] = np.nan

    plot_hits_cond(hits, cond, plotdir)
    plot_res_hist(maps, sky_in, plotdir)
    plot_maps(maps, sky_in, plotdir, diff_range_P=args.range_pol)
    plot_residuals(residuals, plotdir)

    elapsed = time.perf_counter() - tic
    print(f"Elapsed time: {elapsed:.2f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Plot difference maps and histograms for a given run."
    )
    parser.add_argument("dirname", type=utils.dir_path, help="name of directory")
    parser.add_argument("--range-pol", type=int, help="colorbar range for residual Q/U maps")
    parser.add_argument(
        "--hits-percentile",
        type=float,
        default=1,
        help="exclude pixels with less hits than this percentile of the hit map",
    )
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
