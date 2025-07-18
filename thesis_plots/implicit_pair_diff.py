#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="notebook", style="ticks")


JZ_VALIDATION = Path("../jz_validation")
BINNED_HWP = JZ_VALIDATION / "incl/binned/ml"
BINNED_NO_HWP = JZ_VALIDATION / "incl/binned-nohwp"
HWP = JZ_VALIDATION / "incl/correlated"
NO_HWP = JZ_VALIDATION / "incl/correlated_no_hwp"
HWP_NOISELESS = JZ_VALIDATION / "incl/correlated-noiseless"
NO_HWP_NOISELESS = JZ_VALIDATION / "incl/correlated-nohwp-noiseless"

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
    "hwp_noiseless": HWP_NOISELESS,
    "no_hwp": NO_HWP,
    "no_hwp_noiseless": NO_HWP_NOISELESS,
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
        rms = np.nanstd(r)
        amp = 2 * rms
        cartview(
            r,
            sub=(2, 4, plot_num),
            title=f"{hwp_label} {stokes} Residuals",
            min=offset - amp,
            max=offset + amp,
        )

# fig.savefig("implicit_pair_diff.svg", dpi=600, bbox_inches="tight")

# Set up figure comparing residuals between noiseless and non noiseless runs
fig = plt.figure(figsize=(12, 6))

# Compare noiseless vs non-noiseless runs for both HWP and no HWP configurations
configurations = [("hwp", "hwp_noiseless", "HWP"), ("no_hwp", "no_hwp_noiseless", "No HWP")]

stokes = "Q"  # Only compare Q as requested

for j, (noisy_config, noiseless_config, label) in enumerate(configurations):
    # Noisy residuals
    r_noisy = residuals[noisy_config][stokes]
    offset_noisy = np.nanmean(r_noisy)
    rms_noisy = np.nanstd(r_noisy)

    # Noiseless residuals
    r_noiseless = residuals[noiseless_config][stokes]
    offset_noiseless = np.nanmean(r_noiseless)
    rms_noiseless = np.nanstd(r_noiseless)

    # Use the same scale for both plots for fair comparison
    amp = 2 * max(rms_noisy, rms_noiseless)

    # Plot noisy residuals
    plot_num = j * 3 + 1
    cartview(
        r_noisy,
        sub=(2, 3, plot_num),
        title=f"{label} {stokes} (with atmosphere)",
        min=offset_noisy - 2 * rms_noisy,
        max=offset_noisy + 2 * rms_noisy,
    )

    # Plot noiseless residuals
    plot_num = j * 3 + 2
    cartview(
        r_noiseless,
        sub=(2, 3, plot_num),
        title=f"{label} {stokes} (noiseless)",
        min=offset_noiseless - 2 * rms_noiseless,
        max=offset_noiseless + 2 * rms_noiseless,
    )

    # Plot noise increase (difference between noisy and noiseless)
    plot_num = j * 3 + 3
    noise_increase = r_noisy - r_noiseless
    noise_offset = np.nanmean(noise_increase)
    noise_rms = np.nanstd(noise_increase)
    noise_amp = 2 * noise_rms

    cartview(
        noise_increase,
        sub=(2, 3, plot_num),
        title=f"{label} {stokes} difference",
        min=noise_offset - noise_amp,
        max=noise_offset + noise_amp,
    )

fig.savefig("noiseless_comparison.svg", dpi=600, bbox_inches="tight")
