#!/usr/bin/env python3

# Copyright (c) 2019-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs an SO time domain simulation.

You can see the automatically generated command line options with:

    toast_so_sim --help

Or you can dump a config file with all the default values with:

    toast_so_sim --default_toml config.toml

This script contains just comments about what is going on.  For details about all the
options for a specific Operator, see the documentation or use the help() function from
an interactive python session.

"""

import argparse
import datetime
import os
import sys
import traceback

import numpy as np

from astropy import units as u

# Import sotodlib.toast first, since that sets default object names
# to use in toast.
import sotodlib.toast as sotoast

import toast
import toast.ops

from toast.mpi import MPI, Comm
from toast.observation import default_values as defaults

from sotodlib.toast import ops as so_ops
from sotodlib.toast import workflows as wrk

from mappraiser.workflow import setup_mapmaker_mappraiser, mapmaker_mappraiser

# Make sure pixell uses a reliable FFT engine
import pixell.fft

pixell.fft.engine = "fftw"


def simulate_data(job, otherargs, runargs, data):
    log = toast.utils.Logger.get()

    wrk.simulate_atmosphere_signal(job, otherargs, runargs, data)

    # Shortcut if we are only caching the atmosphere.  If this job is only caching
    # (not observing) the atmosphere, then return at this point.
    if job.operators.sim_atmosphere.cache_only or job.operators.sim_atmosphere_coarse.cache_only:
        return

    wrk.simulate_sky_map_signal(job, otherargs, runargs, data)
    wrk.simulate_conviqt_signal(job, otherargs, runargs, data)
    wrk.simulate_scan_synchronous_signal(job, otherargs, runargs, data)
    wrk.simulate_source_signal(job, otherargs, runargs, data)
    wrk.simulate_sso_signal(job, otherargs, runargs, data)
    wrk.simulate_catalog_signal(job, otherargs, runargs, data)
    wrk.simulate_wiregrid_signal(job, otherargs, runargs, data)
    wrk.simulate_stimulator_signal(job, otherargs, runargs, data)
    wrk.simulate_detector_timeconstant(job, otherargs, runargs, data)
    wrk.simulate_mumux_crosstalk(job, otherargs, runargs, data)
    wrk.simulate_detector_noise(job, otherargs, runargs, data)
    wrk.simulate_hwpss_signal(job, otherargs, runargs, data)
    wrk.simulate_detector_yield(job, otherargs, runargs, data)
    wrk.simulate_calibration_error(job, otherargs, runargs, data)
    wrk.simulate_readout_effects(job, otherargs, runargs, data)

    comm = data.comm.comm_world

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After simulating data:  {mem}", comm)

    wrk.save_data_hdf5(job, otherargs, runargs, data)

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After saving data:  {mem}", comm)


def synthesize_noise_and_atm_for_mappraiser(job, data):
    """Add atmosphere and noise buffers together in the mappraiser noise buffer."""
    if not (mappraiser := job.operators.mappraiser).enabled:
        # mappraiser disabled, nothing to do
        return

    # first check if mappraiser's noise data key should exist
    nkey = mappraiser.noise_data
    if (noise_op := job.operators.sim_noise).enabled:
        # the noise simulation operator is enabled, key should exist
        _nkey_exists = True
        if noise_op.det_data != nkey:
            msg = f"Mappraiser noise data key {nkey} does not match noise simulation key {noise_op.det_data}"
            raise RuntimeError(msg)
    else:
        # key should not exist yet
        # Toast will log a warning when we try to copy to an existing key
        _nkey_exists = False

    # Add atmosphere and noise buffers together in the mappraiser noise buffer
    for atm_op in [job.operators.sim_atmosphere, job.operators.sim_atmosphere_coarse]:
        if not atm_op.enabled:
            continue
        akey = atm_op.det_data
        if _nkey_exists:
            toast.ops.Combine(op="add", first=nkey, second=akey, result=nkey).apply(data)
        else:
            toast.ops.Copy(detdata=[(akey, nkey)]).apply(data)
            # noise key was created by the line above
            _nkey_exists = True
            toast.ops.Delete(detdata=[akey]).apply(data)


def reduce_data(job, otherargs, runargs, data):
    log = toast.utils.Logger.get()

    synthesize_noise_and_atm_for_mappraiser(job, data)

    wrk.simple_jumpcorrect(job, otherargs, runargs, data)
    wrk.simple_deglitch(job, otherargs, runargs, data)

    wrk.flag_diff_noise_outliers(job, otherargs, runargs, data)
    wrk.flag_noise_outliers(job, otherargs, runargs, data)
    wrk.deconvolve_detector_timeconstant(job, otherargs, runargs, data)
    wrk.raw_statistics(job, otherargs, runargs, data)

    wrk.filter_hwpss(job, otherargs, runargs, data)
    wrk.filter_common_mode(job, otherargs, runargs, data)
    wrk.filter_ground(job, otherargs, runargs, data)
    wrk.filter_poly1d(job, otherargs, runargs, data)
    wrk.filter_poly2d(job, otherargs, runargs, data)
    wrk.diff_noise_estimation(job, otherargs, runargs, data)
    wrk.noise_estimation(job, otherargs, runargs, data)

    data = wrk.demodulate(job, otherargs, runargs, data)

    wrk.processing_mask(job, otherargs, runargs, data)
    wrk.flag_sso(job, otherargs, runargs, data)
    wrk.hn_map(job, otherargs, runargs, data)
    wrk.cadence_map(job, otherargs, runargs, data)
    wrk.crosslinking_map(job, otherargs, runargs, data)

    if job.operators.splits.enabled:
        wrk.splits(job, otherargs, runargs, data)
    else:
        wrk.mapmaker_ml(job, otherargs, runargs, data)
        wrk.mapmaker(job, otherargs, runargs, data)
        wrk.mapmaker_filterbin(job, otherargs, runargs, data)
        wrk.mapmaker_madam(job, otherargs, runargs, data)
        mapmaker_mappraiser(job, otherargs, runargs, data)
    wrk.filtered_statistics(job, otherargs, runargs, data)

    mem = toast.utils.memreport(msg="(whole node)", comm=data.comm.comm_world, silent=True)
    log.info_rank(f"After reducing data:  {mem}", data.comm.comm_world)


def main():
    env = toast.utils.Environment.get()
    log = toast.utils.Logger.get()
    gt = toast.timing.GlobalTimers.get()
    gt.start("toast_so_sim (total)")
    timer = toast.timing.Timer()
    timer.start()

    # Get optional MPI parameters
    comm, procs, rank = toast.get_world()

    # If the user has not told us to use multiple threads,
    # then just use one.

    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = 1
    log.info_rank(
        f"Executing workflow with {procs} MPI tasks, each with "
        f"{nthread} OpenMP threads at {datetime.datetime.now()}",
        comm,
    )

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"Start of the workflow:  {mem}", comm)

    # Argument parsing
    parser = argparse.ArgumentParser(description="SO simulation pipeline")

    parser.add_argument(
        "--out_dir",
        required=False,
        type=str,
        default="toast_out",
        help="The output directory",
    )
    parser.add_argument(
        "--obsmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each observation separately.",
    )
    parser.add_argument(
        "--detmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each detector separately.",
    )
    parser.add_argument(
        "--intervalmaps",
        required=False,
        default=False,
        action="store_true",
        help="Map each interval separately.",
    )
    parser.add_argument(
        "--zero_loaded_data",
        required=False,
        default=False,
        action="store_true",
        help="Zero out detector data loaded from disk",
    )
    parser.add_argument(
        "--simulate-only",
        required=False,
        default=False,
        action="store_true",
        help="Only simulate the data, do not reduce it",
    )
    parser.add_argument(
        "--single-det",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only use the first detector of each pair",
    )
    parser.add_argument(
        "--scramble-after",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Pass the gainscrambler to mappraiser to be applied after noise estimation"
    )

    # The operators and templates we want to configure from the command line
    # or a parameter file.

    operators = list()
    templates = list()

    wrk.setup_load_or_simulate_observing(parser, operators)

    wrk.setup_simulate_atmosphere_signal(operators)
    wrk.setup_simulate_sky_map_signal(operators)
    wrk.setup_simulate_conviqt_signal(operators)
    wrk.setup_simulate_scan_synchronous_signal(operators)
    wrk.setup_simulate_source_signal(operators)
    wrk.setup_simulate_sso_signal(operators)
    wrk.setup_simulate_catalog_signal(operators)
    wrk.setup_simulate_wiregrid_signal(operators)
    wrk.setup_simulate_stimulator_signal(operators)
    wrk.setup_simulate_detector_timeconstant(operators)
    wrk.setup_simulate_mumux_crosstalk(operators)
    wrk.setup_simulate_detector_noise(operators)
    wrk.setup_simulate_hwpss_signal(operators)
    wrk.setup_simulate_detector_yield(operators)
    wrk.setup_simulate_calibration_error(operators)
    wrk.setup_simulate_readout_effects(operators)
    wrk.setup_save_data_hdf5(operators)

    wrk.setup_simple_jumpcorrect(operators)
    wrk.setup_simple_deglitch(operators)

    wrk.setup_flag_diff_noise_outliers(operators)
    wrk.setup_flag_noise_outliers(operators)
    wrk.setup_deconvolve_detector_timeconstant(operators)
    wrk.setup_raw_statistics(operators)

    wrk.setup_filter_hwpss(operators)
    wrk.setup_filter_common_mode(operators)
    wrk.setup_filter_ground(operators)
    wrk.setup_filter_poly1d(operators)
    wrk.setup_filter_poly2d(operators)
    wrk.setup_noise_estimation(operators)

    wrk.setup_demodulate(operators)

    wrk.setup_processing_mask(operators)
    wrk.setup_flag_sso(operators)
    wrk.setup_hn_map(operators)
    wrk.setup_cadence_map(operators)
    wrk.setup_crosslinking_map(operators)

    wrk.setup_mapmaker_ml(operators)
    wrk.setup_mapmaker(operators, templates)
    wrk.setup_mapmaker_filterbin(operators)
    wrk.setup_mapmaker_madam(operators)
    wrk.setup_splits(operators)
    wrk.setup_filtered_statistics(operators)

    setup_mapmaker_mappraiser(parser, operators)

    job, config, otherargs, runargs = wrk.setup_job(
        parser=parser, operators=operators, templates=templates
    )

    # Create our output directory
    if comm is None or comm.rank == 0:
        if not os.path.isdir(otherargs.out_dir):
            os.makedirs(otherargs.out_dir, exist_ok=True)

    # Log the config that was actually used at runtime.
    outlog = os.path.join(otherargs.out_dir, "config_log.toml")
    toast.config.dump_toml(outlog, config, comm=comm)

    # If this is a dry run, exit
    if otherargs.dry_run:
        log.info_rank("Dry-run complete", comm=comm)
        return

    data = wrk.load_or_simulate_observing(job, otherargs, runargs, comm)

    simulate_data(job, otherargs, runargs, data)

    if not job.operators.sim_atmosphere.cache_only and not otherargs.simulate_only:
        # Reduce the data
        reduce_data(job, otherargs, runargs, data)

    # Collect optional timing information
    alltimers = toast.timing.gather_timers(comm=comm)
    if data.comm.world_rank == 0:
        out = os.path.join(otherargs.out_dir, "timing")
        toast.timing.dump(alltimers, out)

    log.info_rank("Workflow completed in", comm=comm, timer=timer)


def cli():
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()


if __name__ == "__main__":
    cli()
