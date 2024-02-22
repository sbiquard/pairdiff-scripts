#!/usr/bin/env python3

import os
import sys
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
        nest=True,
    )
    hp.gnomview(
        cond,
        title="Inverse condition number map",
        sub=[1, 2, 2],
        rot=rot,
        xsize=xsize,
        notext=False,
        cmap="bwr",
        nest=True,
    )
    fig.savefig(os.path.join(savedir, "hits.png"))


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
    fig.savefig(os.path.join(savedir, "diff_histograms.png"))


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
            nest=True,
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
                nest=True,
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
                nest=True,
                unit=unit,
            )
    fig.savefig(os.path.join(savedir, "maps.png"))


def process(args):
    # start timer
    tic = time.perf_counter()

    # unpack args
    dirname, ref = args

    # create folder for the plots
    savedir = os.path.join(dirname, "figs", ref)
    os.makedirs(savedir, exist_ok=True)

    # read data
    iqu, maps_out = utils.read_maps(dirname, ref=ref)
    hits, cond = utils.read_hits_cond(dirname, ref=ref)
    sky_in = 1e6 * hp.fitsfunc.read_map(
        "ffp10_lensed_scl_100_nside0512.fits", field=None, nest=True
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

    elapsed = time.perf_counter() - tic
    return ref, elapsed


def main(dirname, refs):
    # Use up to 4 cpus
    ncpu = min(len(refs), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=ncpu) as pool:
        print(f"Using {ncpu} CPU")
        for ref, elapsed in pool.imap_unordered(
            process, zip((dirname for _ in refs), refs)
        ):
            print(f"Processed ref '{ref}' in {elapsed:.3f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: ./display_maps.py dirname [refs]")
    else:
        dirname = sys.argv[1]
        dirname = os.path.join(os.getcwd(), dirname)
        if len(sys.argv) < 3:
            refs = [utils.get_last_ref(dirname)]
        else:
            refs = sys.argv[2:]
        print(f"Begin work in directory '{dirname}' with {len(refs)} ref(s)")
        main(dirname, refs)
