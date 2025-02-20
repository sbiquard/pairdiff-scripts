#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import toast
from toast.ops import LoadHDF5
from utils import get_all_runs


def rel_diff_sqr(a: float, b: float):
    return (a**2 - b**2) / (a**2 + b**2)


def pairwise(s: Iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    a = iter(s)
    return zip(a, a)


def load_epsilon_dist(src: Path):
    data = toast.Data()
    LoadHDF5(volume=str(src)).apply(data)

    def epsilon(ob):
        model = ob["noise_model"]
        return [
            rel_diff_sqr(model.NET(det_A), model.NET(det_B))
            for det_A, det_B in pairwise(model.detectors)
        ]

    return np.concatenate([epsilon(ob) for ob in data.obs])


def transfer(src_root: Path, dest_root: Path):
    for run in get_all_runs(src_root):
        data_dir = run / "var_noise_model_data"
        if not data_dir.exists():
            print(f"Skipping {run}")
            continue
        dist = load_epsilon_dist(data_dir)
        dest = dest_root / run.relative_to(src_root)
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "epsilon_dist", dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="source root")
    parser.add_argument("dest", type=Path, help="destination root")
    args = parser.parse_args()
    transfer(args.src, args.dest)


if __name__ == "__main__":
    main()
