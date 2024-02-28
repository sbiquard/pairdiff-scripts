import pathlib
import numpy as np
import healpy as hp
import toml

import functools
import time


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


def get_last_ref(dirname):
    run = pathlib.Path(dirname)
    params = toml.load(run / "mappraiser_args_log.toml")
    ref = params.get("ref", "run0")  # last ref logged
    return ref


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

    # load the output maps
    mapI = None
    if iqu:
        mapI = 1e6 * hp.fitsfunc.read_map(str(run / f"mapI_{ref}.fits"), field=None)
    mapQ = 1e6 * hp.fitsfunc.read_map(str(run / f"mapQ_{ref}.fits"), field=None)
    mapU = 1e6 * hp.fitsfunc.read_map(str(run / f"mapU_{ref}.fits"), field=None)

    return iqu, (mapI, mapQ, mapU)


def read_hits_cond(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # load hits and condition number maps
    run = pathlib.Path(dirname)
    hits = hp.fitsfunc.read_map(
        str(run / f"Hits_{ref}.fits"), field=None, dtype=np.int32
    )
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


def read_mask(fname="out/mask_apo.fits"):
    file = pathlib.Path(fname)
    if file.exists():
        mask = hp.read_map(str(file))
        return mask
    else:
        msg = f"{file} not found -- run get_mask_apo.py to generate it"
        raise FileNotFoundError(msg)
