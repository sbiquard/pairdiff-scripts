#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.visualization import hist
import multiprocessing
import time
import functools

import utils

LONRA = [-95, 135]
LATRA = [-70, -10]


def plot_hits_cond(hits, cond, savedir, cmap="bwr"):
    def plot(m, title):
        hp.cartview(m, norm="hist", title=title, lonra=LONRA, latra=LATRA, cmap=cmap)

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

    for k, v in maps.items():
        resid[k] = sky_in[convert[k]] - maps[k]

    fig, ax = plt.subplots(figsize=(7, 7))

    for stokes, residual in resid.items():
        residual = residual[~np.isnan(residual)]
        hist(
            residual,
            bins="scott",
            label=f"{stokes} ; {np.mean(residual):.2e} +/- {np.std(residual):.2e} $\\mu K$",
            histtype="step",
            ax=ax,
        )

    fig.suptitle("Histograms of residuals")
    ax.set_xlabel("$\\mu K_{CMB}$")
    ax.grid(True)
    ax.legend()
    fig.savefig(savedir / "diff_histograms.png")
    plt.close(fig)


def plot_maps(
    maps,
    sky_in,
    savedir,
    cmap="bwr",
    map_range_T=500,
    map_range_P=20,
):
    nrow, ncol = 3, 3
    fig = plt.figure(figsize=(8 * ncol, 4 * nrow))

    unit = "$\\mu K_{CMB}$"
    convert = {"I": 0, "Q": 1, "U": 2}

    for i, stokes in enumerate(convert):
        map_range = map_range_T if i == 0 else map_range_P

        # Plot input sky
        hp.cartview(
            sky_in[i],
            title=f"Input {stokes}",
            lonra=LONRA,
            latra=LATRA,
            sub=[nrow, ncol, 1 + 3 * i],
            notext=False,
            min=-map_range,
            max=map_range,
            cmap=cmap,
            unit=unit,
        )

        if stokes in maps:
            # Plot reconstructed map
            hp.cartview(
                maps[stokes],
                title=f"Reconstructed {stokes} map",
                lonra=LONRA,
                latra=LATRA,
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
            hp.cartview(
                diff,
                title=f"Difference {stokes}",
                lonra=LONRA,
                latra=LATRA,
                sub=[nrow, ncol, 1 + 3 * i + 2],
                notext=False,
                cmap="bwr",
                min=offset - amp,
                max=offset + amp,
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


def process(ref, dirname):
    # start timer
    tic = time.perf_counter()

    # create folder for the plots
    run = pathlib.Path(dirname)
    savedir = run / "plots" / ref
    savedir.mkdir(parents=True, exist_ok=True)

    # read data
    maps = utils.read_maps(dirname, ref=ref)
    hits, cond = utils.read_hits_cond(dirname, ref=ref)
    residuals = utils.read_residuals(dirname, ref=ref)
    sky_in = utils.read_input_sky()

    # define a mask for pixels outside the solved patch
    mask = hits < 1
    for m in maps.values():
        m[mask] = np.nan
    cond[mask] = np.nan

    plot_hits_cond(hits, cond, savedir)
    plot_res_hist(maps, sky_in, savedir)
    plot_maps(maps, sky_in, savedir, map_range_P=10)
    plot_residuals(residuals, savedir)

    elapsed = time.perf_counter() - tic
    return ref, elapsed


def main(dirname, refs, verbose):
    if refs is None:
        refs = [utils.get_last_ref(dirname)]
    if verbose:
        print(f"Process {len(refs)} ref(s) in '{dirname}'")

    # Use up to 4 cpus
    ncpu = min(len(refs), 4)
    with multiprocessing.Pool(processes=ncpu) as pool:
        if verbose:
            print(f"Using {ncpu} CPU")
        partial_func = functools.partial(process, dirname=dirname)
        for ref, elapsed in pool.imap_unordered(partial_func, refs):
            if verbose:
                print(f"Processed ref '{ref}' in {elapsed:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot difference maps and histograms for a given run."
    )
    parser.add_argument("dirname", type=str, help="name of directory")
    parser.add_argument("--refs", nargs="*", type=str, help="refs to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    args = parser.parse_args()
    main(args.dirname, args.refs, args.verbose)
