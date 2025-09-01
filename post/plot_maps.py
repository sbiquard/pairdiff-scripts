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
    maps_mirror = utils.read_maps(run, ref=ref, mirror=True)
    hits, cond = utils.read_hits_cond(run, ref=ref)
    hits_mirror, cond_mirror = utils.read_hits_cond(run, ref=ref, mirror=True)
    residuals = utils.read_residuals(run, ref=ref)
    sky_in = utils.read_input_sky(name=args.sky)

    # define a mask for pixels with less hits than the given percentile
    thresh = np.max(hits) * args.hits_frac
    mask = hits < thresh
    for m in maps.values():
        m[mask] = np.nan
    cond[mask] = np.nan

    # same for mirror maps
    if maps_mirror is not None:
        for m in maps_mirror.values():
            m[hits_mirror == 0] = np.nan
        cond_mirror[hits_mirror == 0] = np.nan

    plotting.plot_hits_cond(
        hits, cond, plotdir, mirror_hits=hits_mirror, mirror_cond=cond_mirror, s2=args.s2
    )
    plotting.plot_res_hist(maps, sky_in, plotdir)
    plotting.plot_maps(
        maps, sky_in, plotdir, diff_range_P=args.diff_range_P, mirrors=maps_mirror, s2=args.s2
    )
    plotting.plot_residuals(residuals, plotdir)

    elapsed = time.perf_counter() - tic
    return run, elapsed


def main():
    parser = argparse.ArgumentParser(description="Produce plots for all runs under a given root")
    parser.add_argument("roots", type=utils.dir_path, nargs="+", help="List of run directories")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--ncpu", type=int, default=4)
    parser.add_argument("--diff-range-P", type=int)
    parser.add_argument("--hits-frac", type=float, default=0.1)
    parser.add_argument("--sky", type=str)
    parser.add_argument("-p", "--pattern", type=str, default=None)
    parser.add_argument("-s2", action="store_true")
    args = parser.parse_args()

    runs = list(utils.get_all_runs(args.roots, pattern=args.pattern))
    if len(runs) == 0:
        print(f"No runs found under {args.roots!r} with pattern {args.pattern!r}")
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
