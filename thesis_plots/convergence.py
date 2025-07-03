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
# plt.title("Convergence of Residuals")
plt.plot(iterations_ml, rel_residuals_ml, label="full IQU")
plt.plot(iterations_pd, rel_residuals_pd, label="pair differencing")
plt.xlabel("Iteration number")
plt.ylabel("Relative residual reduction")
plt.yscale("log")  # Log scale often better for residual plots
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("convergence.png", bbox_inches="tight", dpi=600)
