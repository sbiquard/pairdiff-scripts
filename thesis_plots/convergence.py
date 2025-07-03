#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_logs(log_file):
    residuals_squared = []
    with open(log_file, "r") as f:
        # Parse each line of the log file
        for line in f:
            line = line.strip()
            if line.startswith("k ="):
                parts = line.split(",")
                r_squared = float(parts[1].split("=")[1].strip())
                residuals_squared.append(r_squared)
    return np.array(residuals_squared)


# broken :/
# residuals_ml = np.loadtxt(JZ / "leak/incl_turnarounds/diff_gains_with_instr/ml/residuals_run0.dat")
# residuals_pd = np.loadtxt(JZ / "leak/incl_turnarounds/diff_gains_with_instr/pd/residuals_run0.dat")
log_ml = Path("../jz_out/leak/incl_turnarounds/diff_gains_with_instr/ml/run.log")
log_pd = Path("../jz_out/leak/incl_turnarounds/diff_gains_with_instr/pd/run.log")

# Calculate relative residuals by dividing by first residual
residuals_squared_ml = parse_logs(log_ml)
residuals_squared_pd = parse_logs(log_pd)
rel_residuals_ml = np.sqrt(residuals_squared_ml / residuals_squared_ml[0])
rel_residuals_pd = np.sqrt(residuals_squared_pd / residuals_squared_pd[0])

# Create iteration indices
iterations_ml = np.arange(rel_residuals_ml.size)
iterations_pd = np.arange(rel_residuals_pd.size)

# Create the plot
fig, ax = plt.subplots()
ax.plot(iterations_ml, rel_residuals_ml, label="full IQU")
ax.plot(iterations_pd, rel_residuals_pd, label="pair differencing")
ax.axhline(1e-8, color="k", ls="--", label="convergence criterion")
ax.set(
    xlabel="Iteration number",
    ylabel="Relative residual reduction",
    yscale="log",  # Log scale often better for residual plots
)
ax.grid(visible=True)
ax.legend()

# Create an inset axes for zooming on the PD curve
axins = ax.inset_axes((0.15, 0.2, 0.35, 0.3))
axins.plot(iterations_pd, rel_residuals_pd, color=ax.lines[1].get_color())
axins.axhline(1e-8, color="k", ls="--")
axins.set_yscale("log")
axins.grid(True)

# Adjust inset axes view to focus on PD curve
niter_pd = len(iterations_pd)
axins.set_xlim(-1, niter_pd)
axins.set_xticks(np.arange(0, niter_pd, 4))
# axins.grid(False)

# ax.indicate_inset_zoom(axins, edgecolor="black")

# fig.tight_layout()
fig.savefig("convergence.png", bbox_inches="tight", dpi=600)
