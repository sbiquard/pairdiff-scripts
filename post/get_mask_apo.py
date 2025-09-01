#!/usr/bin/env python3

import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import spectrum
import utils


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="create and save an apodized mask (for power spectrum computations)",
    )
    parser.add_argument("location", type=Path, help="where to find the hit map")
    parser.add_argument(
        "--hits-fraction",
        dest="hits_frac",
        default=0.1,
        type=float,
        help="maximum hits fraction threshold",
    )
    parser.add_argument(
        "--aposize",
        default=10.0,
        type=float,
        help="scale of apodization in degrees",
    )
    parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        default="mask_apo",
        help="file name under which to save the mask",
    )
    args = parser.parse_args()

    # Read the baseline hits map
    print(f"Read hit map from '{args.location}'")
    hits, _ = utils.read_hits_cond(args.location)

    # Cut under a minimum hit count and apodize
    print(f"Compute mask with hits/hits_max > {args.hits_frac:%} apodized over {args.aposize} degrees")
    mask = spectrum.get_mask_apo(hits, args.hits_frac, args.aposize)

    # Save the mask
    maskname = Path("out") / args.filename
    fname_save = maskname.with_suffix(".fits")
    print(f"Save apodized mask under '{fname_save}'")
    hp.write_map(fname_save, mask, overwrite=True)

    # Plot the mask
    fname_plot = maskname.with_suffix(".png")
    print(f"Save mask plot under '{fname_plot}'")
    hp.projview(mask, title=f"Apodized mask (hits_frac={args.hits_frac}, aposize={args.aposize})")
    plt.savefig(fname_plot)


if __name__ == "__main__":
    main()
