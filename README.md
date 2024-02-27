# Simulation files for pair differencing evaluation

Scripts and parameter files used in the pair differencing project.

## General

The simulation comprises only the SAT 90 GHz frequency band and spans one observing year.

We simulate the following data:

* noise : atmosphere, instrumental noise
* cmb

We only simulate the first calendar day of each month.

## Software

* [TOAST 3.0.0a20.dev61](https://github.com/hpc4cmb/toast/tree/c2ac5a7806ecffeacd42ba47e64322076e824fb8)
* [sotodlib 0.5.0+534.g468bd7f7](https://github.com/simonsobs/sotodlib/tree/468bd7f75eddb2b324a40a0c17917056221b21f4)
* [midapack/mappraiser](https://github.com/B3Dcmb/midapack/tree/gaps)

## Files in this directory

Setup

* `get_defaults.sh` : Use `so_sim_mappraiser` to generate a default parameter file for reference
* `sat.toml` : Master parameter file for the `so_sim_mappraiser` workflow
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
* `plot.py` : Produce difference maps and histograms for a given run
* `spectrum.py` : Power spectrum routines
