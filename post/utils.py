import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import toml


def dir_path(string):
    if os.path.isdir(string):
        return string
    raise NotADirectoryError(string)


SKIP_DIRS = ["plots", "spectra", "atm_cache"]


def get_all_runs(root, exclude=SKIP_DIRS):
    root = Path(root)
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


def contains_log(run: Path):
    for _ in run.glob("*.log"):
        return True
    return False


def get_last_ref(dirname):
    run = Path(dirname)

    try:
        # try json first
        logfile = (run / "mappraiser_args_log").with_suffix(".json")
        with logfile.open() as config:
            params = json.load(config)
    except FileNotFoundError:
        # fallback to toml
        logfile = (run / "config_log").with_suffix(".toml")
        with logfile.open() as config:
            params = toml.load(config)
    return params.get("ref", "run0")


def is_complete(run: Path):
    ref = get_last_ref(run)
    fnames = (
        f"Cond_{ref}.fits",
        f"Hits_{ref}.fits",
        f"mapQ_{ref}.fits",
        f"mapU_{ref}.fits",
        f"residuals_{ref}.dat",
    )

    for fname in fnames:
        if not (run / fname).exists():
            return False

    return True


def read_input_sky(field=None):
    filename = Path(__file__).parents[1] / "ffp10_lensed_scl_100_nside0512.fits"
    return 1e6 * hp.fitsfunc.read_map(filename, field=field)


def read_maps(dirname, ref=None, mask=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # read logged param file
    run = Path(dirname)
    params = toml.load(run / "config_log.toml")
    params = params["operators"]["mappraiser"]

    # do we have iqu maps or just qu?
    iqu = not (params["pair_diff"]) or params["estimate_spin_zero"]

    # read the output maps and put them in a dict
    maps = {}
    if iqu:
        maps["I"] = 1e6 * hp.fitsfunc.read_map(str(run / f"mapI_{ref}.fits"), field=None)
    maps["Q"] = 1e6 * hp.fitsfunc.read_map(str(run / f"mapQ_{ref}.fits"), field=None)
    maps["U"] = 1e6 * hp.fitsfunc.read_map(str(run / f"mapU_{ref}.fits"), field=None)

    if mask is None:
        return maps
    return {k: np.where(mask, v, hp.UNSEEN) for k, v in maps.items()}


def read_hits_cond(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # load hits and condition number maps
    run = Path(dirname)
    hits = hp.fitsfunc.read_map(str(run / f"Hits_{ref}.fits"), field=None, dtype=np.int32)
    cond = hp.fitsfunc.read_map(str(run / f"Cond_{ref}.fits"), field=None)

    return hits, cond


def read_hits(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # load hits and condition number maps
    run = Path(dirname)
    return hp.fitsfunc.read_map(str(run / f"Hits_{ref}.fits"), field=None, dtype=np.int32)


def read_residuals(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # read residuals file
    run = Path(dirname)
    fname = f"residuals_{ref}.dat"
    data = np.loadtxt(run / fname, skiprows=1, usecols=1)
    return data


def read_mask(fname="mask_apo"):
    file = (Path(__file__).parents[1] / "out" / fname).with_suffix(".fits")
    if file.exists():
        mask = hp.read_map(str(file))
        return mask
    msg = f"{file} not found -- run `get_mask_apo.py --out {fname}` to generate it"
    raise FileNotFoundError(msg)
