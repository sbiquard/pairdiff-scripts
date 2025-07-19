#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import toast
from scipy.signal import welch
from toast.ops import LoadHDF5

sns.set_theme(context="notebook", style="ticks")

JZ = Path("../jz_out")
OBS = JZ / "sample_obs_data/data/obs_DEC-050..-030_RA+000.000..+011.613-0-0_ST1_1735144136.h5"

# Load sample atmosphere data and compute PSD
data = toast.Data()
LoadHDF5(
    files=[str(OBS)],
    detdata=["signal", "atm", "atm_coarse", "noise", "pixels", "weights"],
    shared=["times", "flags", "boresight_radec"],
).apply(data)

ob = data.obs[0]
freq, Pxx = welch(ob.detdata["atm"][0], fs=37, nperseg=1024)
NET = 245.1e-6
eta = NET**2
# print(f"Regularization: {eta * 1e12} µK²/Hz")

JZ_VALIDATION = Path("../jz_validation")
HWP = JZ_VALIDATION / "incl/correlated-regularized"
NO_HWP = JZ_VALIDATION / "incl/correlated-regularized-nohwp"

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

# First subplot for PSD
fig, ax = plt.subplots(figsize=(5, 5))
ax.loglog(freq, 1e12 * (Pxx), label="Atmosphere")
ax.loglog(freq, 1e12 * (Pxx + eta), label="Regularized")
ax.set(xlabel="Frequency [Hz]", ylabel="Power Spectral Density [µK²/Hz]")
ax.legend()
fig.savefig("regularization_psd.svg", dpi=600, bbox_inches="tight")

# Second plot with four residual plots
fig = plt.figure(figsize=(6, 5))

# Loop through both Q and U for both HWP configurations
plot_positions = {
    ("hwp", "Q"): (0, 0),
    ("hwp", "U"): (1, 0),
    ("no_hwp", "Q"): (0, 1),
    ("no_hwp", "U"): (1, 1),
}

for hwp_config in ["hwp", "no_hwp"]:
    hwp_label = "HWP" if hwp_config == "hwp" else "No HWP"

    for stokes in ["Q", "U"]:
        r = residuals[hwp_config][stokes]
        offset = np.nanmean(r)

        # Use fixed range for HWP residuals as requested
        if hwp_config == "hwp":
            min_val = -1e-6
            max_val = 1e-6
        else:
            min_val = -1e-2
            max_val = +1e-2

        row, col = plot_positions[(hwp_config, stokes)]

        cartview(
            r,
            sub=(2, 2, row * 2 + col + 1),
            title=f"{hwp_label} {stokes} Regularized",
            min=min_val,
            max=max_val,
        )

fig.savefig("regularization_maps.svg", dpi=600, bbox_inches="tight")
