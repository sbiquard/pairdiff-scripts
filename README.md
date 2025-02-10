# Simulation files for pair differencing evaluation

Scripts and parameter files used in the pair differencing project.

## General

The simulation comprises only the SAT 90 GHz frequency band and spans one observing year.

We simulate the following data:

* noise: atmosphere, instrumental noise
* cmb

We only simulate the first calendar day of each month.

## Software

The software packages used are
[TOAST 3](https://github.com/hpc4cmb/toast/tree/toast3),
[sotodlib](https://github.com/simonsobs/sotodlib)
and [mappraiser](https://github.com/B3Dcmb/midapack/tree/gaps-maxL).
They are provided as submodules in the `extern` folder so that the exact setup can be reproduced easily.

The simulation worfklow, `so_mappraiser.py`, is a modified version of the `toast_so_sim.py` script in sotodlib.

## Files in this directory

Unless otherwise noted, all scripts should be run from the root of the repository.

Setup

* `get_defaults.sh` : Use `so_mappraiser.py` to generate a default parameter file for reference
* `sat.toml` : Master parameter file for the `so_mappraiser.py` workflow
* `sat.par`, `atm.par` : Sets of command line parameters for the workflow
* `ffp10_lensed_scl_100_nside0512.fits` : Input map to be observed during simulation

Schedule files

* `schedules/schedule.01.south.txt` : Schedule file
* `schedules/schedule.small.txt` : Truncated schedule file for laptop tests
* `schedules/schedule.opti.txt` : Schedule file with a single scan

Tests (laptop: truncated schedule, decimated focal plane)

* `tests/opti` : Evaluate the optimality of pair-differencing compared to maximum-likelihood (single observation)
  * `run.white.uniform.sh` : all detector _pairs_ have the same white noise level (but not detectors in a pair)
  * `run.white.variable.sh` : all detectors have different white noise levels
  * `run.one_over_f.sh` : all detectors have different 1/f noise parameters
  * `run.atm.sh` : in addition to variable instrumental noise, simulate atmosphere
* `tests/syst` : Evaluate the impact of systematic effects on the pair-differencing approach
  * `run.atm.cache.sh` : simulate and cache the atmosphere simulation
  * `run.baseline.sh` : run the baseline configuration (ideal case)
  * `run.gains.constant.sh` : run with gain errors which are the same for all detector pairs

Execution (Jean-Zay: full schedule)

* `slurm/run.atm.cache.slurm` : Simulate and cache the atmosphere simulation
* `slurm/get_sample_data.slurm` : Get sample observation data for testing
* `slurm/opti/*`: Run the optimality tests

Post-processing

* `post/compute_spectra.py` : Compute and save power spectra for all runs
* `post/get_input_spectra.py` : Compute and save power spectra of input map
* `post/get_mask_apo.py` : Create and save a mask (requires [NaMaster](https://namaster.readthedocs.io))
* `post/plot_maps_all.py` : Plot difference maps and histograms for all runs in a root directory
* `post/plot_maps.py` : Produce difference maps and histograms for a given run
* `post/plot_spectra.py` : Plot power spectra recursively for all runs in a root directory
* `post/spectrum.py` : Power spectrum routines
* `slurm/run.spectra.slurm` : Job script to compute power spectra
