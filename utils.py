import functools
import json
import os
import pathlib
import time

import healpy as hp
import numpy as np
import toml


def timer(func):
    """Simple timer decorator"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds", flush=True)
        return value

    return wrapper_timer


# Utility routines


SKIP_DIRS = ["plots", "spectra", "atm_cache"]


def dir_path(string):
    if os.path.isdir(string):
        return string
    raise NotADirectoryError(string)


def get_all_runs(root, exclude=SKIP_DIRS):
    root = pathlib.Path(root)
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


def get_last_ref(dirname, logfile_suffix=".json"):
    run = pathlib.Path(dirname)
    logfile = (run / "mappraiser_args_log").with_suffix(logfile_suffix)
    with logfile.open() as config:
        params = json.load(config)
        ref = params.get("ref", "run0")  # last ref logged
    return ref


def is_complete(run: pathlib.Path):
    try:
        ref = get_last_ref(run)
    except FileNotFoundError:
        try:
            # try with .toml extension
            ref = get_last_ref(run, ".toml")
        except FileNotFoundError:
            return False

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
    sky = 1e6 * hp.fitsfunc.read_map("ffp10_lensed_scl_100_nside0512.fits", field=field)
    return sky


def read_maps(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # read logged param file
    run = pathlib.Path(dirname)
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

    return maps


def read_hits_cond(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # load hits and condition number maps
    run = pathlib.Path(dirname)
    hits = hp.fitsfunc.read_map(str(run / f"Hits_{ref}.fits"), field=None, dtype=np.int32)
    cond = hp.fitsfunc.read_map(str(run / f"Cond_{ref}.fits"), field=None)

    return hits, cond


def read_residuals(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # read residuals file
    run = pathlib.Path(dirname)
    fname = f"residuals_{ref}.dat"
    data = np.loadtxt(run / fname, skiprows=1, usecols=1)
    return data


def read_mask(fname="mask_apo"):
    file = pathlib.Path("out") / f"{fname}.fits"
    if file.exists():
        mask = hp.read_map(str(file))
        return mask
    msg = f"{file} not found -- run `get_mask_apo.py --out {fname}` to generate it"
    raise FileNotFoundError(msg)
