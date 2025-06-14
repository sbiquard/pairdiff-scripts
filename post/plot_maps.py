#!/usr/bin/env python3

import argparse
import multiprocessing
import time
from functools import partial

import numpy as np
import plotting
import utils


def process(run, args):
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
    sky_in = utils.read_input_sky(name=args.sky)

    # define a mask for pixels with less hits than the given percentile
    thresh = np.percentile(hits[hits > 0], args.hits_percentile)
    mask = hits < thresh
    for m in maps.values():
        m[mask] = np.nan
    cond[mask] = np.nan

    plotting.plot_hits_cond(hits, cond, plotdir)
    plotting.plot_res_hist(maps, sky_in, plotdir)
    plotting.plot_maps(maps, sky_in, plotdir, diff_range_P=args.diff_range_P)
    plotting.plot_residuals(residuals, plotdir)

    elapsed = time.perf_counter() - tic
    return run, elapsed


def main():
    parser = argparse.ArgumentParser(description="Produce plots for all runs under a given root")
    parser.add_argument("root", type=utils.dir_path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--ncpu", type=int, default=4)
    parser.add_argument("--diff-range-P", type=int)
    parser.add_argument("--hits-percentile", type=float, default=0)
    parser.add_argument("--sky", type=str)
    parser.add_argument("-p", "--pattern", type=str, default="*")
    args = parser.parse_args()

    runs = list(utils.get_all_runs(args.root, pattern=args.pattern))
    if len(runs) == 0:
        print(f"No runs found under {args.root!r} with pattern {args.pattern!r}")
        return

    ncpu = args.ncpu if args.ncpu > 0 else multiprocessing.cpu_count()
    ncpu = min(ncpu, len(runs))

    with multiprocessing.Pool(processes=ncpu) as pool:
        if args.verbose:
            print(f"Using {ncpu} CPU")
        for run, elapsed in pool.imap_unordered(partial(process, args=args), runs):
            if elapsed is None:
                print(f"Could not plot maps for '{run}' (missing files)")
                continue
            if args.verbose:
                print(f"Processed '{run}' in {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
