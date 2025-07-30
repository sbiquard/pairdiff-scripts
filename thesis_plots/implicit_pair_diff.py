#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="notebook", style="ticks")


JZ_VALIDATION = Path("../jz_validation")

# implicit pair diff plot
BINNED_HWP = JZ_VALIDATION / "incl/binned/ml"
BINNED_NO_HWP = JZ_VALIDATION / "incl/binned-nohwp"
HWP = JZ_VALIDATION / "incl/correlated"
NO_HWP = JZ_VALIDATION / "incl/correlated-nohwp"

# noiseless plots
HWP_REGUL = JZ_VALIDATION / "incl/correlated-regularized"
NO_HWP_REGUL = JZ_VALIDATION / "incl/correlated-regularized-nohwp"
HWP_NOISELESS = JZ_VALIDATION / "incl/correlated-regularized-noiseless"
NO_HWP_NOISELESS = JZ_VALIDATION / "incl/correlated-regularized-nohwp-noiseless"
HWP_NOISELESS_POLAR = JZ_VALIDATION / "incl/correlated-regularized-noiseless-polar"
NO_HWP_NOISELESS_POLAR = JZ_VALIDATION / "incl/correlated-regularized-nohwp-noiseless-polar"

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
    "hwp_regul": HWP_REGUL,
    "hwp_noiseless": HWP_NOISELESS,
    "no_hwp": NO_HWP,
    "no_hwp_regul": NO_HWP_REGUL,
    "no_hwp_noiseless": NO_HWP_NOISELESS,
    "binned_hwp": BINNED_HWP,
    "binned_no_hwp": BINNED_NO_HWP,
    "hwp_noiseless_polar": HWP_NOISELESS_POLAR,
    "no_hwp_noiseless_polar": NO_HWP_NOISELESS_POLAR,
}

# Read hitmaps and maps for each data source
hitmaps = {key: read_hitmap(path) for key, path in data_sources.items()}
maps = {key: read_maps(path) for key, path in data_sources.items()}

# Plot residuals for Q and U of the hwp and no_hwp runs
cartview = partial(
    hp.cartview, unit=r"$\mu K$", xsize=5000, latra=[-70, -10], lonra=[-30, 90], cmap="bwr"
)

# Calculate residuals
threshold = 100
residuals = {}
for run in data_sources.keys():
    residuals[run] = {}
    residuals[run]["Q"] = np.where(hitmaps[run] > threshold, maps[run][:, 1] - SKY[1], np.nan)
    residuals[run]["U"] = np.where(hitmaps[run] > threshold, maps[run][:, 2] - SKY[2], np.nan)

# Set up figure
fig = plt.figure(figsize=(12, 5))

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

        # Use custom color range based on the type of run
        if "binned" in hwp_config:
            # For binned runs, use +/- 1e-10
            min_val = -1e-10
            max_val = +1e-10
        elif hwp_config == "hwp":
            # For correlated HWP, use +/- 5
            min_val = -5
            max_val = +5
        else:
            # For other configurations, use the original 2*rms
            rms = np.nanstd(r)
            min_val = offset - 2 * rms
            max_val = offset + 2 * rms

        cartview(
            r,
            sub=(2, 4, plot_num),
            title=f"{hwp_label} {stokes} Residuals",
            min=min_val,
            max=max_val,
        )

fig.savefig("implicit_pair_diff.svg", dpi=600, bbox_inches="tight")


# Set up figure comparing residuals between noiseless and non noiseless runs
fig = plt.figure(figsize=(10, 5))  # Wider to accommodate third column

# Compare noiseless vs non-noiseless runs for both HWP and no HWP configurations
configurations = [
    ("hwp_regul", "hwp_noiseless", "hwp_noiseless_polar", "HWP"),
    ("no_hwp_regul", "no_hwp_noiseless", "no_hwp_noiseless_polar", "No HWP"),
]

stokes = "Q"  # Only compare Q as requested

for j, (noisy_config, noiseless_config, polar_noiseless_config, label) in enumerate(configurations):
    # Noisy residuals
    r_noisy = residuals[noisy_config][stokes]
    offset_noisy = np.nanmean(r_noisy)
    rms_noisy = np.nanstd(r_noisy)

    # Noiseless residuals
    r_noiseless = residuals[noiseless_config][stokes]
    offset_noiseless = np.nanmean(r_noiseless)
    rms_noiseless = np.nanstd(r_noiseless)

    # Noiseless polar residuals
    r_noiseless_polar = residuals[polar_noiseless_config][stokes]
    offset_noiseless_polar = np.nanmean(r_noiseless_polar)
    rms_noiseless_polar = np.nanstd(r_noiseless_polar)

    # Plot noisy residuals
    plot_num = j * 3 + 1
    min_val = offset_noisy - 2 * rms_noisy
    max_val = offset_noisy + 2 * rms_noisy

    cartview(
        r_noisy,
        sub=(2, 3, plot_num),
        title=f"{label} {stokes} (with atmosphere)",
        min=min_val,
        max=max_val,
    )

    # Plot noiseless residuals
    plot_num = j * 3 + 2
    min_val = offset_noiseless - 2 * rms_noiseless
    max_val = offset_noiseless + 2 * rms_noiseless

    cartview(
        r_noiseless,
        sub=(2, 3, plot_num),
        title=f"{label} {stokes} (noiseless)",
        min=min_val,
        max=max_val,
    )

    # Plot noiseless polar residuals
    plot_num = j * 3 + 3
    min_val = offset_noiseless_polar - 2 * rms_noiseless_polar
    max_val = offset_noiseless_polar + 2 * rms_noiseless_polar

    cartview(
        r_noiseless_polar,
        sub=(2, 3, plot_num),
        title=f"{label} {stokes} (polarization only)",
        min=min_val,
        max=max_val,
    )

fig.savefig("noiseless_regul_comparison.svg", dpi=600, bbox_inches="tight")
