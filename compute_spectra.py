#!/usr/bin/env python3

import argparse
import numpy as np
import multiprocessing
import functools
import time

import utils
import spectrum
import plot_spectra


def add_arguments(parser):
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite previously computed spectra",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument(
        "-dl",
        "--delta-ell",
        dest="delta_ell",
        type=int,
        default=5,
        help="size of ell bins (default: 5)",
    )
    parser.add_argument(
        "-n",
        "--ncpu",
        type=int,
        default=4,
        help="number of CPUs to use (default: 4)",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="save a plot of the computed spectra"
    )


def process(run, delta_ell: int, overwrite: bool):
    # start timer
    tic = time.perf_counter()

    # check if directory is complete
    if not utils.is_complete(run):
        return run, None

    # create directory if needed
    cldir = run / "spectra"
    cldir.mkdir(exist_ok=True)

    # check if there is something to do
    # TODO: handle several refs
    full_cl_name = cldir / "full_cl.npz"
    noise_cl_name = cldir / "noise_cl.npz"
    if full_cl_name.exists() and noise_cl_name.exists() and not overwrite:
        return run, 0.0

    # read maps
    maps = utils.read_maps(run)
    if "I" in maps:
        sky = utils.read_input_sky()
        maps = np.array([maps["I"], maps["Q"], maps["U"]])
    else:
        # only polarization
        sky = utils.read_input_sky(field=(1, 2))
        maps = np.array([maps["Q"], maps["U"]])

    # read mask and get binning scheme
    mask_apo = utils.read_mask()
    binning = spectrum.get_binning(delta_ell)

    # compute the spectra
    full_cl = spectrum.compute_spectra(maps, mask_apo, binning)
    noise_cl = spectrum.compute_spectra(maps - sky, mask_apo, binning)

    # save to npz files
    np.savez(full_cl_name, **full_cl, ell_arr=binning.get_effective_ells())
    np.savez(noise_cl_name, **noise_cl, ell_arr=binning.get_effective_ells())

    elapsed = time.perf_counter() - tic
    return run, elapsed


def plot(run):
    plot_spectra.process(run)
    return run


def main(args):
    runs = list(utils.get_all_runs("out"))
    # hp.projview(mask_apo)
    # plt.show()

    if args.ncpu > 0:
        ncpu = args.ncpu
    else:
        ncpu = multiprocessing.cpu_count()

    # Don't use more CPUs than runs to process
    ncpu = min(ncpu, len(runs))

    with multiprocessing.Pool(processes=ncpu) as pool:
        if args.verbose:
            print(f"Using {ncpu} CPU")
        partial_func = functools.partial(
            process, delta_ell=args.delta_ell, overwrite=args.overwrite
        )
        runs_complete = list()
        for run, elapsed in pool.imap_unordered(partial_func, runs):
            if elapsed is None:
                print(f"Could not compute spectra for '{run}' (missing files)")
                continue
            runs_complete.append(run)
            if args.verbose:
                if elapsed == 0.0:
                    print(f"Skipped '{run}' (nothing to do)")
                else:
                    print(f"Processed '{run}' in {elapsed:.3f} seconds")
        if args.plot:
            # produce plots if needed
            for run in pool.imap_unordered(plot, runs_complete):
                if args.verbose:
                    print(f"Produced plot for '{run}'")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save power spectra for all runs.",
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
