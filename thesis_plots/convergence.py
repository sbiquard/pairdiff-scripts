#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="paper", style="ticks")


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


logs = {}
logs["ml"] = Path("../jz_out/leak/incl_turnarounds/diff_gains_with_instr/ml/run.log")
logs["pd"] = Path("../jz_out/leak/incl_turnarounds/diff_gains_with_instr/pd/run.log")
logs["ml_nohwp"] = Path("../jz_out/leak/incl_turnarounds/diff_gains_with_instr_nohwp/ml/run.log")
logs["pd_nohwp"] = Path("../jz_out/leak/incl_turnarounds/diff_gains_with_instr_nohwp/pd/run.log")

# Calculate relative residuals by dividing by first residual
relative_residuals = {}
iterations = {}
for key, log in logs.items():
    if not log.exists():
        raise FileNotFoundError(f"Log file {log} does not exist.")
    res_squared = parse_logs(log)
    relative_residuals[key] = np.sqrt(res_squared / res_squared[0])
    iterations[key] = np.arange(len(res_squared))

# Create the plot
sns.set_palette("Paired")
fig, ax = plt.subplots()
ax.plot(iterations["ml"], relative_residuals["ml"], label="IQU")
ax.plot(iterations["ml_nohwp"], relative_residuals["ml_nohwp"], label="IQU (no HWP)")
ax.plot(iterations["pd"], relative_residuals["pd"], label="pair diff")
ax.plot(iterations["pd_nohwp"], relative_residuals["pd_nohwp"], label="pair diff (no HWP)")
ax.axhline(1e-8, color="k", ls="--", label="convergence criterion")
ax.set(
    xlabel="Iteration number",
    ylabel="Relative residual reduction",
    yscale="log",  # Log scale often better for residual plots
)
# ax.set_ylim(bottom=1e-10)
ax.grid(visible=True)
ax.legend()

# Create an inset axes for zooming on the PD curve
axins = ax.inset_axes((0.2, 0.2, 0.4, 0.3))
axins.plot(iterations["pd"], relative_residuals["pd"], color=ax.lines[2].get_color())
axins.plot(iterations["pd_nohwp"], relative_residuals["pd_nohwp"], color=ax.lines[3].get_color())
axins.axhline(1e-8, color="k", ls="--")
axins.set_yscale("log")
axins.grid(True)

# Adjust inset axes view to focus on PD curve
niter = iterations["pd_nohwp"].size
# axins.set_xlim(-1, niter)
# axins.set_xticks(np.arange(0, niter, 4))
# axins.grid(False)

# ax.indicate_inset_zoom(axins, edgecolor="black")

# fig.tight_layout()
fig.savefig("convergence.svg", bbox_inches="tight")
