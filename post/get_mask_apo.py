#!/usr/bin/env python3

import argparse

import healpy as hp
import matplotlib.pyplot as plt
import spectrum
import utils


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="create and save an apodized mask (for power spectrum computations)",
    )
    parser.add_argument(
        "--basedir",
        default="out/baseline",
        help="directory where to find the hit map",
    )
    parser.add_argument(
        "--min-hits",
        dest="min_hits",
        default=1_000,
        type=int,
        help="hit count threshold (default: 1000)",
    )
    parser.add_argument(
        "--aposize",
        default=10.0,
        type=float,
        help="scale of apodization in degrees (default: 10 deg)",
    )
    parser.add_argument(
        "--out",
        dest="file_name",
        default="mask_apo",
        help="file name under which to save the mask (must not already exist)",
    )
    args = parser.parse_args()

    # Read the baseline hits map
    print(f"Read hit map from '{args.basedir}'")
    hits, _ = utils.read_hits_cond(args.basedir)

    # Cut under a minimum hit count and apodize
    print(f"Compute mask with hits > {args.min_hits:_} apodized over {args.aposize} degrees")
    mask = spectrum.get_mask_apo(hits, args.min_hits, args.aposize)

    # Save the mask
    fname_save = f"out/{args.file_name}.fits"
    print(f"Save apodized mask under '{fname_save}'")
    hp.write_map(fname_save, mask, overwrite=False)

    # Plot the mask
    fname_plot = f"out/{args.file_name}.png"
    print(f"Save mask plot under '{fname_plot}'")
    hp.projview(mask, title=f"Apodized mask (minhits={args.min_hits}, aposize={args.aposize})")
    plt.savefig(fname_plot)


if __name__ == "__main__":
    main()
