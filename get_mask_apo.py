#!/usr/bin/env python3

import sys
import healpy as hp
import matplotlib.pyplot as plt

import utils
import spectrum


def f(min_hits):
    # Read the baseline hits map
    hits, _ = utils.read_hits_cond("out/baseline")

    # Cut under a minimum hit count and apodize
    if min_hits is not None:
        mask, b = spectrum.get_mask_apo_and_binning(hits, min_hits=min_hits)
    else:
        mask, b = spectrum.get_mask_apo_and_binning(hits)

    # Save the mask
    hp.write_map("out/mask_apo.fits", mask, overwrite=True)

    # Plot the mask
    hp.projview(mask, title="Apodized mask")
    plt.savefig("out/mask_apo.png")


if __name__ == "__main__":
    min_hits = int(sys.argv[1]) if len(sys.argv) >= 2 else None
    f(min_hits)
