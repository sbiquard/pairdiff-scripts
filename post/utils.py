import json
import os
import re
from pathlib import Path

import healpy as hp
import numpy as np
import toml


def dir_path(string):
    if os.path.isdir(string):
        return string
    raise NotADirectoryError(string)


SKIP_DIRS = ["plots", "spectra", "atm_cache", "noise_psd_fit", "var_noise_model_data"]


def get_all_runs(root: list[str | Path] | str | Path, exclude=None, pattern=None):
    """
    Get all runs in the given directory or directories and their subdirectories.

    Args:
        root: The root directory or list of root directories to search
        exclude: List of directory names to exclude
        pattern: Regex pattern to match directory names
    """
    if exclude is None:
        exclude = SKIP_DIRS

    # Convert a single directory path to a list
    if not isinstance(root, list):
        roots = [Path(root)]
    else:
        roots = [Path(r) for r in root]

    # Process each root directory
    for root_dir in roots:
        if root_dir.name in ("ml", "pd"):
            continue

        # Check if root directory itself is a valid run or if it matches pattern for special dirs
        matches_pattern = pattern is None or re.search(pattern, root_dir.name)

        if matches_pattern:
            if contains_log(root_dir):
                # If the root directory contains a log file, yield it as a run
                yield root_dir

            # Check for special directories 'ml' and 'pd'
            for special_dir in ["ml", "pd"]:
                special_path = root_dir / special_dir
                if special_path.is_dir():
                    yield special_path

        # Recursively explore contents
        for item in root_dir.iterdir():
            if not item.is_dir() or item.name in SKIP_DIRS:
                # skip files and excluded directories
                continue

            yield from get_all_runs(item, exclude=exclude, pattern=pattern)


def contains_log(run: Path) -> bool:
    for _ in run.glob("*.log"):
        return True
    return False


def get_last_ref(dirname) -> None | str:
    run = Path(dirname)
    if not contains_log(run):
        return None

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
    if ref is None:
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


def read_input_sky(field=None, name=None):
    if name is None:
        name = "ffp10_lensed_scl_100_nside0512.fits"
    filename = Path(__file__).parents[1] / name
    return 1e6 * hp.read_map(filename, field=field, dtype=np.float64)


def read_maps(dirname, ref=None, mask=None, mirror=False):
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
    mirror_suffix = "_mirror" if mirror else ""

    try:
        if iqu:
            maps["I"] = 1e6 * hp.read_map(
                str(run / f"mapI_{ref}{mirror_suffix}.fits"), field=None, dtype=np.float64
            )
        maps["Q"] = 1e6 * hp.read_map(
            str(run / f"mapQ_{ref}{mirror_suffix}.fits"), field=None, dtype=np.float64
        )
        maps["U"] = 1e6 * hp.read_map(
            str(run / f"mapU_{ref}{mirror_suffix}.fits"), field=None, dtype=np.float64
        )
    except FileNotFoundError:
        return None

    if mask is None:
        return maps
    return {k: np.where(mask, v, hp.UNSEEN) for k, v in maps.items()}


def read_hits_cond(dirname, ref=None, mirror=False):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # load hits and condition number maps
    run = Path(dirname)
    mirror_suffix = "_mirror" if mirror else ""
    try:
        hits = hp.read_map(str(run / f"Hits_{ref}{mirror_suffix}.fits"), field=None, dtype=np.int32)
        cond = hp.read_map(
            str(run / f"Cond_{ref}{mirror_suffix}.fits"), field=None, dtype=np.float64
        )
    except FileNotFoundError:
        return None, None

    return hits, cond


def read_hits(dirname, ref=None):
    # ref of the run
    if ref is None:
        ref = get_last_ref(dirname)

    # load hits and condition number maps
    run = Path(dirname)
    return hp.read_map(str(run / f"Hits_{ref}.fits"), field=None, dtype=np.int32)


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
