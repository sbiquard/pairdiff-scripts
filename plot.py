#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.visualization import hist
import multiprocessing
import time

import utils


def plot_hits_cond(hits, cond, savedir, xsize=4000, rot=[15, -40]):
    fig = plt.figure(figsize=(10, 4))
    hp.gnomview(
        hits,
        title="Hits map",
        sub=[1, 2, 1],
        rot=rot,
        xsize=xsize,
        notext=False,
        cmap="bwr",
    )
    hp.gnomview(
        cond,
        title="Inverse condition number map",
        sub=[1, 2, 2],
        rot=rot,
        xsize=xsize,
        notext=False,
        cmap="bwr",
    )
    fig.savefig(savedir / "hits.png")


def plot_res_hist(maps_out, iqu, sky_in, savedir):
    if iqu:
        aI = sky_in[0] - maps_out[0]
        aI = aI[~np.isnan(aI)]

    aQ = sky_in[1] - maps_out[1]
    aQ = aQ[~np.isnan(aQ)]

    aU = sky_in[2] - maps_out[2]
    aU = aU[~np.isnan(aU)]

    fig, ax = plt.subplots(figsize=(7, 7))

    if iqu:
        for i in range(3):
            stokes = list("TQU")[i]
            residual = (aI, aQ, aU)[i]
            hist(
                residual,
                bins="scott",
                label=f"{stokes} ; {np.mean(residual):.2e} +/- {np.std(residual):.2e} $\\mu K$",
                histtype="step",
                ax=ax,
            )
    else:
        for i in range(2):
            stokes = list("QU")[i]
            residual = (aQ, aU)[i]
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


def plot_maps(
    maps_out,
    iqu,
    sky_in,
    savedir,
    xsize=4000,
    rot=[15, -40],
    map_range_T=500,
    map_range_P=20,
):
    nrow, ncol = 3, 3
    fig = plt.figure(figsize=(4 * ncol, 3 * nrow))

    unit = "$\\mu K_{CMB}$"

    for i in range(3):
        stokes = list("TQU")[i]
        map_range = map_range_T if i == 0 else map_range_P

        # Input map
        hp.gnomview(
            sky_in[i],
            rot=rot,
            xsize=xsize,
            sub=[nrow, ncol, 1 + 3 * i],
            title=f"Input {stokes}",
            notext=False,
            min=-map_range,
            max=map_range,
            cmap="bwr",
            unit=unit,
        )

        # Reconstructed map
        if iqu or (i > 0):
            hp.gnomview(
                maps_out[i],
                rot=rot,
                xsize=xsize,
                sub=[nrow, ncol, 1 + 3 * i + 1],
                title=f"Reconstructed {stokes} map",
                notext=False,
                min=-map_range,
                max=map_range,
                cmap="bwr",
                unit=unit,
            )
            m = sky_in[i] - maps_out[i]
            offset = np.nanmedian(m)
            rms = np.nanstd(m)
            amp = 2 * rms
            hp.gnomview(
                sky_in[i] - maps_out[i],
                rot=rot,
                xsize=xsize,
                sub=[nrow, ncol, 1 + 3 * i + 2],
                title=f"Difference {stokes}",
                notext=False,
                cmap="bwr",
                min=offset - amp,
                max=offset + amp,
                unit=unit,
            )
    fig.savefig(savedir / "maps.png")


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
    fig.savefig(savedir / "pcg_residuals.png")


def process(args):
    # start timer
    tic = time.perf_counter()

    # unpack args
    dirname, ref = args
    run = pathlib.Path(dirname)

    # create folder for the plots
    savedir = run / "plots" / ref
    savedir.mkdir(parents=True, exist_ok=True)

    # read data
    iqu, maps_out = utils.read_maps(dirname, ref=ref)
    hits, cond = utils.read_hits_cond(dirname, ref=ref)
    residuals = utils.read_residuals(dirname, ref=ref)
    sky_in = 1e6 * hp.fitsfunc.read_map(
        "ffp10_lensed_scl_100_nside0512.fits", field=None
    )

    # define a mask for pixels outside the solved patch
    mask = hits < 1
    for m in maps_out:
        if m is not None:
            m[mask] = np.nan
    cond[mask] = np.nan

    xsize = 2500
    rot = [50, -40]  # SO south patch

    plot_hits_cond(hits, cond, savedir, xsize=xsize, rot=rot)
    plot_res_hist(maps_out, iqu, sky_in, savedir)
    plot_maps(
        maps_out,
        iqu,
        sky_in,
        savedir,
        xsize=xsize,
        rot=rot,
        map_range_P=10,
    )
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
        for ref, elapsed in pool.imap_unordered(
            process, zip((dirname for _ in refs), refs)
        ):
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
