#!/usr/bin/env python3

import sys
import healpy as hp
import matplotlib.pyplot as plt

import utils
import spectrum


def f(min_hits):
    # Read the baseline hits map
    base_dir = "out/baseline"
    print(f"Read hit map from '{base_dir}'")
    hits, _ = utils.read_hits_cond(base_dir)

    # Cut under a minimum hit count and apodize
    print(f"Compute apodized mask ({min_hits=})")
    if min_hits is not None:
        mask, b = spectrum.get_mask_apo_and_binning(hits, min_hits=min_hits)
    else:
        mask, b = spectrum.get_mask_apo_and_binning(hits)

    # Save the mask
    fname_save = "out/mask_apo.fits"
    print(f"Save apodized mask under '{fname_save}'")
    hp.write_map(fname_save, mask, overwrite=True)

    # Plot the mask
    fname_plot = "out/mask_apo.png"
    print(f"Save map plot under '{fname_plot}'")
    hp.projview(mask, title="Apodized mask")
    plt.savefig(fname_plot)


if __name__ == "__main__":
    min_hits = int(sys.argv[1]) if len(sys.argv) >= 2 else None
    f(min_hits)
