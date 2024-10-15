# Simulation files for pair differencing evaluation

Scripts and parameter files used in the pair differencing project.

## General

The simulation comprises only the SAT 90 GHz frequency band and spans one observing year.

We simulate the following data:

* noise: atmosphere, instrumental noise
* cmb

We only simulate the first calendar day of each month.

## Software

The software packages used are [TOAST 3](https://github.com/hpc4cmb/toast/tree/toast3), [sotodlib](https://github.com/simonsobs/sotodlib) and [midapack/mappraiser](https://github.com/B3Dcmb/midapack/tree/gaps).
They are provided as submodules of this repository so that the exact setup can be reproduced by others.

## Files in this directory

Setup

* `get_defaults.sh` : Use `toast_so_mappraiser.py` to generate a default parameter file for reference
* `sat.toml` : Master parameter file for the `toast_so_mappraiser.py` workflow
* `schedule.01.south.txt` : Schedule file
* `schedule.small.txt` : Truncated schedule file for laptop tests
* `ffp10_lensed_scl_100_nside0512.fits` : Input map to be observed during simulation

Tests (laptop: truncated schedule, decimated focal plane)

* `run.atm.cache.sh` : Simulate and cache the atmosphere simulation
* `run.baseline.sh` : Run the baseline configuration (ideal case)
* `run.gains.constant.sh` : Run with gain errors which are the same for all detector pairs

Execution (Jean-Zay: full schedule)

* `run.atm.cache.slurm` : Simulate and cache the atmosphere simulation
* `run.baseline.slurm` : Run the baseline configuration (ideal case)
* `run.gains.constant.slurm` : Run with gain errors which are the same for all detector pairs
* `run.gains.random.slurm` : Run with Gaussian distributed gain errors

Post-processing

* `utils.py` : Some utility routines
* `plot_maps.py` : Produce difference maps and histograms for a given run
* `spectrum.py` : Power spectrum routines
* `get_mask_apo.py` : Create and save a mask (requires [NaMaster](https://namaster.readthedocs.io))
* `compute_spectra.py` : Compute and save power spectra for all runs
* `run.spectra.slurm` : Job script to compute power spectra
