#!/usr/bin/env python3

import argparse
import numpy as np
import multiprocessing
import time

import utils
import plot_maps


def add_arguments(parser):
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument(
        "-n",
        "--ncpu",
        type=int,
        default=4,
        help="number of CPUs to use (default: 4)",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite previously computed spectra",
    )


def process(run):
    # start timer
    tic = time.perf_counter()

    # check if directory is complete
    if not utils.is_complete(run):
        return run, None

    # create directory if needed
    plotdir = run / "plots"
    plotdir.mkdir(exist_ok=True)

    # get last ref
    ref = utils.get_last_ref(run)

    # read data
    maps = utils.read_maps(run, ref=ref)
    hits, cond = utils.read_hits_cond(run, ref=ref)
    residuals = utils.read_residuals(run, ref=ref)
    sky_in = utils.read_input_sky()

    # define a mask for pixels outside the solved patch
    mask = hits < 1
    for m in maps.values():
        m[mask] = np.nan
    cond[mask] = np.nan

    plot_maps.plot_hits_cond(hits, cond, plotdir)
    plot_maps.plot_res_hist(maps, sky_in, plotdir)
    plot_maps.plot_maps(maps, sky_in, plotdir, map_range_P=10)
    plot_maps.plot_residuals(residuals, plotdir)

    elapsed = time.perf_counter() - tic
    return run, elapsed


def main(args):
    runs = list(utils.get_all_runs("out"))

    if args.ncpu > 0:
        ncpu = args.ncpu
    else:
        ncpu = multiprocessing.cpu_count()

    # Don't use more CPUs than runs to process
    ncpu = min(ncpu, len(runs))

    with multiprocessing.Pool(processes=ncpu) as pool:
        if args.verbose:
            print(f"Using {ncpu} CPU")
        for run, elapsed in pool.imap_unordered(process, runs):
            if elapsed is None:
                print(f"Could not plot maps for '{run}' (missing files)")
                continue
            if args.verbose:
                print(f"Processed '{run}' in {elapsed:.3f} seconds")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce plots of output maps for all runs.",
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
