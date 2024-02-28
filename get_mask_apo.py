#!/usr/bin/env python3

import argparse
import healpy as hp
import matplotlib.pyplot as plt

import utils
import spectrum


def add_arguments(parser):
    parser.add_argument(
        "--min-hits",
        dest="min_hits",
        default=1000,
        type=int,
        help="hit count threshold (default: 1000)",
    )
    parser.add_argument(
        "--aposize",
        default=10.0,
        type=float,
        help="scale of apodization in degrees (default: 10 deg)",
    )


def main(args):
    # Read the baseline hits map
    base_dir = "out/baseline"
    print(f"Read hit map from '{base_dir}'")
    hits, _ = utils.read_hits_cond(base_dir)

    # Cut under a minimum hit count and apodize
    print("Compute apodized mask")
    mask = spectrum.get_mask_apo(hits, args.min_hits, args.aposize)

    # Save the mask
    fname_save = "out/mask_apo.fits"
    print(f"Save apodized mask under '{fname_save}'")
    hp.write_map(fname_save, mask, overwrite=True)

    # Plot the mask
    fname_plot = "out/mask_apo.png"
    print(f"Save mask plot under '{fname_plot}'")
    hp.projview(mask, title="Apodized mask")
    plt.savefig(fname_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create and save an apodized mask (for power spectrum computations)",
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
