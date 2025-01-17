#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import utils
import spectrum
import plot_spectra


@utils.timer
def main(args):
    # create directory if needed
    out = pathlib.Path("out")
    out.mkdir(exist_ok=True)

    clname = out / "input_cells.npz"
    if clname.exists():
        # spectrum already computed
        cl = np.load(clname)
    else:
        # read input sky
        sky = utils.read_input_sky()

        # read mask and get binning scheme
        mask_apo = utils.read_mask()
        binning = spectrum.get_binning(args.delta_ell)

        # compute the spectra
        cl = spectrum.compute_spectra(sky, mask_apo, binning)
        cl["ell_arr"] = binning.get_effective_ells()

        # save to npz file
        np.savez(clname, **cl)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Power spectrum of input sky")
    has_T = plot_spectra.plot(cl, axs[0])
    has_T = plot_spectra.plot(cl, axs[1], dl=True)
    plot_spectra.decorate(axs[0], has_T=has_T)
    plot_spectra.decorate(axs[1], has_T=has_T, dl=True)
    plt.savefig(out / "input_spectrum.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save spectra of the input sky.")
    parser.add_argument(
        "-dl",
        "--delta-ell",
        dest="delta_ell",
        type=int,
        default=5,
        help="size of ell bins (default: 5)",
    )
    args = parser.parse_args()
    main(args)
