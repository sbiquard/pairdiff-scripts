import functools
import time
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import utils

LONRA = [-95, 135]
LATRA = [-70, -10]

cartview = functools.partial(hp.cartview, lonra=LONRA, latra=LATRA)
cartview_sky = functools.partial(cartview, cmap="bwr", unit=r"$\mu K_{CMB}$")


def plot_hits_cond(hits, cond, savedir, mirror_hits=None, mirror_cond=None):
    def plot(m, title, mirror_m=None):
        if mirror_m is None:
            cartview(m, title=title)
            return
        cartview(m, title=title, sub=211)
        cartview(mirror_m, title="Mirror", sub=212)

    # Plot hits
    plot(hits, "Hits map", mirror_m=mirror_hits)
    plt.savefig(savedir / "hits.png", bbox_inches="tight")
    plt.close()

    # Plot cond
    plot(cond, "Inverse condition number map", mirror_m=mirror_cond)
    plt.savefig(savedir / "cond.png", bbox_inches="tight")
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
            ax.hist(
                residual,
                bins="auto",
                label=f"{stoke} ; {np.mean(residual):.2e} +/- {np.std(residual):.2e} $\\mu K$",
                histtype="step",
            )

        fig.suptitle(f"Histograms of {stokes!r} residuals")
        ax.set_xlabel("$\\mu K_{CMB}$")
        ax.grid(True)
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(savedir / f"diff_histograms_{stokes}.png", bbox_inches="tight")
        plt.close(fig)

    # plot I separately
    if "I" in maps:
        plot_hist_stoke("I")
    plot_hist_stoke("QU")


def plot_maps(
    maps,
    sky_in,
    savedir: Path,
    map_range_T: float = 500,
    map_range_P: float = 15,
    diff_range_T: float | None = None,
    diff_range_P: float | None = None,
    mirrors=None,
):
    nrow, ncol = 3, (3 if mirrors is None else 4)
    fig = plt.figure(figsize=(8 * ncol, 4 * nrow))

    for i, stokes in enumerate("IQU"):
        map_range = map_range_T if i == 0 else map_range_P

        # Plot input sky
        cartview_sky(
            sky_in[i],
            title=f"Input {stokes}",
            sub=[nrow, ncol, ncol * i + 1],
            min=-map_range,
            max=map_range,
        )

        if stokes in maps:
            # Plot reconstructed map
            cartview_sky(
                maps[stokes],
                title=f"Reconstructed {stokes} map",
                sub=[nrow, ncol, ncol * i + 2],
                min=-map_range,
                max=map_range,
            )

            # Plot difference map
            diff_range = diff_range_T if i == 0 else diff_range_P
            diff = maps[stokes] - sky_in[i]
            offset = np.nanmean(diff)
            rms = np.nanstd(diff)
            amp = 2 * rms
            cartview_sky(
                diff,
                title=f"Difference {stokes}",
                sub=[nrow, ncol, ncol * i + 3],
                min=-(diff_range or -(offset - amp)),
                max=diff_range or (offset + amp),
            )

        if mirrors is not None and stokes in mirrors:
            # Plot mirror map
            cartview_sky(
                mirrors[stokes],
                title=f"Mirror {stokes}",
                sub=[nrow, ncol, ncol * i + 4],
            )

    fig.savefig(savedir / "maps.png", bbox_inches="tight")
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
    fig.savefig(savedir / "pcg_residuals.png", bbox_inches="tight")
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
