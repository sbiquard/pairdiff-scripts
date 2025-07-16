#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="paper", style="ticks")


JZ_VALIDATION = Path("../jz_validation")
BINNED_HWP = JZ_VALIDATION / "incl/binned/ml"
BINNED_NO_HWP = JZ_VALIDATION / "incl/binned-nohwp"
HWP = JZ_VALIDATION / "incl/correlated-cond-noiseless"
NO_HWP = JZ_VALIDATION / "incl/correlated-cond-nohwp-noiseless"

SKY = 1e6 * hp.read_map(
    Path("..") / "ffp10_lensed_scl_100_nside0512.fits",
    field=[0, 1, 2],
    dtype=np.float64,
)


def read_hitmap(path):
    """Read and scale a hitmap from the specified path."""
    return hp.read_map(path / "Hits_run0.fits", dtype=np.int32)


def read_maps(path):
    """Read and stack I, Q, U maps from the specified path."""
    return np.column_stack(
        [1e6 * hp.read_map(path / f"map{stoke}_run0.fits", dtype=np.float64) for stoke in "IQU"]
    )


# Define data sources
data_sources = {
    "hwp": HWP,
    "no_hwp": NO_HWP,
    "binned_hwp": BINNED_HWP,
    "binned_no_hwp": BINNED_NO_HWP,
}

# Read hitmaps and maps for each data source
hitmaps = {key: read_hitmap(path) for key, path in data_sources.items()}
maps = {key: read_maps(path) for key, path in data_sources.items()}

# Plot residuals for Q and U of the hwp and no_hwp runs
cartview = partial(
    hp.cartview, unit=r"$\mu K$", xsize=5000, latra=[-70, -10], lonra=[-30, 90], cmap="bwr"
)

# Calculate residuals
residuals = {}
for run in ["hwp", "no_hwp"]:
    residuals[run] = {}
    residuals[run]["Q"] = np.where(hitmaps[run] > 0, maps[run][:, 1] - SKY[1], np.nan)
    residuals[run]["U"] = np.where(hitmaps[run] > 0, maps[run][:, 2] - SKY[2], np.nan)

# Set up figure
fig = plt.figure(figsize=(12, 5))

# Calculate residuals for binned cases
for run in ["binned_hwp", "binned_no_hwp"]:
    residuals[run] = {}
    residuals[run]["Q"] = np.where(hitmaps[run] > 0, maps[run][:, 1] - SKY[1], np.nan)
    residuals[run]["U"] = np.where(hitmaps[run] > 0, maps[run][:, 2] - SKY[2], np.nan)

# Use double loop to plot residuals for both stokes parameters and HWP configurations
for i, stokes in enumerate(["Q", "U"]):
    for j, hwp_config in enumerate(["binned_hwp", "binned_no_hwp", "hwp", "no_hwp"]):
        plot_num = i * 4 + j + 1  # Calculate subplot position
        if hwp_config == "hwp":
            hwp_label = "HWP"
        elif hwp_config == "no_hwp":
            hwp_label = "No HWP"
        elif hwp_config == "binned_hwp":
            hwp_label = "Binned HWP"
        else:
            hwp_label = "Binned No HWP"

        r = residuals[hwp_config][stokes]
        offset = np.nanmean(r)
        rms = np.nanstd(r)
        amp = 2 * rms
        cartview(
            r,
            sub=(2, 4, plot_num),
            title=f"{hwp_label} {stokes} Residuals",
            min=offset - amp,
            max=offset + amp,
        )

fig.savefig("implicit_pair_diff.svg", dpi=600, bbox_inches="tight")
