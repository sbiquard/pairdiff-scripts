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
* [sotodlib 0.5.0+433.g73d2ba16](https://github.com/simonsobs/sotodlib/tree/1d1560c2a414f2d5cc2552bdf866837efebfd862)
* [midapack/mappraiser](https://github.com/B3Dcmb/midapack/tree/gaps)

## Files in this directory

Setup

* `get_defaults.sh` : Use `so_sim_mappraiser` to generate a default parameter file for reference
* `sat.toml` : Master parameter file for the `so_sim_mappraiser` workflow
* `schedule.01.south.txt` : Schedule file
* `ffp10_lensed_scl_100_nside0512.fits` : Input map to be observed during simulation
