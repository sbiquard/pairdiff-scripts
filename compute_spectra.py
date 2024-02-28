#!/usr/bin/env python3

import argparse
import pathlib
import healpy as hp
# import matplotlib.pyplot as plt

import utils
import spectrum


SKIP_DIRS = ["plots"]


def get_all_runs(root: pathlib.Path, exclude=SKIP_DIRS):
    for item in root.iterdir():
        if not item.is_dir():
            # skip files
            continue
        if item.name in exclude:
            # skip excluded directories
            continue
        if contains_log(item):
            # only yield if the directory contains a log file
            # i.e. it contains the output of a run
            yield item
        # recursively explore
        yield from get_all_runs(item)


def contains_log(run: pathlib.Path):
    for _ in run.glob("*.log"):
        return True
    return False


def add_arguments(parser):
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite previously computed spectra",
    )
    parser.add_argument(
        "-dl", "--delta-ell", dest="delta_ell", default=5, help="size of ell bins"
    )


def main(args):
    runs = get_all_runs(pathlib.Path("out"))
    mask_apo = utils.read_mask()
    binning = spectrum.get_binning(args.delta_ell)
    # hp.projview(mask_apo)
    # plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="compute and save power spectra for all runs",
    )
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
