#!/usr/bin/env python3

# Copyright (c) 2019-2023 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""
This script runs an SO time domain simulation and reduces data to maps using Mappraiser.

You can see the automatically generated command line options with:

    toast_so_mappraiser.py --help

Or you can dump a config file with all the default values with:

    toast_so_mappraiser.py --default_toml config.toml

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
import pixell.fft
import sotodlib.toast as sotoast  # import before toast to set the correct names
import toast
import toast.ops
import toast.utils
from astropy import units as u
from sotodlib.toast import ops as so_ops
from sotodlib.toast import workflows as so_wrk
from toast.mpi import MPI, Comm
from toast.observation import default_values as defaults

from pymappraiser.workflow import setup_mapmaker_mappraiser, mapmaker_mappraiser

# Make sure pixell uses a reliable FFT engine
pixell.fft.engine = "fftw"


def simulate_data(job, otherargs, runargs, comm):
    log = toast.utils.Logger.get()
    job_ops = job.operators

    if job_ops.sim_ground.enabled:
        data = so_wrk.simulate_observing(job, otherargs, runargs, comm)
        if data is None:
            raise RuntimeError("Failed to simulate observing")
    else:
        group_size = so_wrk.reduction_group_size(job, runargs, comm)
        toast_comm = toast.Comm(world=comm, groupsize=group_size)
        data = toast.Data(comm=toast_comm)
        # Load data from all formats
        so_wrk.load_data_hdf5(job, otherargs, runargs, data)
        so_wrk.load_data_books(job, otherargs, runargs, data)
        so_wrk.load_data_context(job, otherargs, runargs, data)
        # optionally zero out
        if otherargs.zero_loaded_data:
            toast.ops.Reset(detdata=[defaults.signal])

    so_wrk.select_pointing(job, otherargs, runargs, data)
    so_wrk.simple_noise_models(job, otherargs, runargs, data)
    so_wrk.simulate_atmosphere_signal(job, otherargs, runargs, data)

    # Shortcut if we are only caching the atmosphere.  If this job is only caching
    # (not observing) the atmosphere, then return at this point.
    if job.operators.sim_atmosphere.cache_only:
        return data

    so_wrk.simulate_sky_map_signal(job, otherargs, runargs, data)
    so_wrk.simulate_conviqt_signal(job, otherargs, runargs, data)
    so_wrk.simulate_detector_noise(job, otherargs, runargs, data)
    so_wrk.simulate_calibration_error(job, otherargs, runargs, data)

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After simulating data:  {mem}", comm)

    so_wrk.save_data_hdf5(job, otherargs, runargs, data)

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"After saving data:  {mem}", comm)

    return data


def reduce_data(job, otherargs, runargs, data):
    log = toast.utils.Logger.get()

    # add atmosphere to noise buffer
    if job.operators.sim_atmosphere.enabled:
        atm_name = job.operators.sim_atmosphere.det_data
        noise_name = job.operators.sim_noise.det_data
        if atm_name != noise_name:
            msg = f"Adding atmosphere '{atm_name}' to noise buffer '{noise_name}'"
            log.info_rank(msg, data.comm.comm_world)
            toast.ops.arithmetic.Combine(
                op="add",
                first=noise_name,
                second=atm_name,
                result=noise_name,
            ).apply(data)

    so_wrk.flag_noise_outliers(job, otherargs, runargs, data)
    so_wrk.noise_estimation(job, otherargs, runargs, data)
    so_wrk.raw_statistics(job, otherargs, runargs, data)

    # use mappraiser as mapmaker
    mapmaker_mappraiser(job, otherargs, runargs, data)

    mem = toast.utils.memreport(
        msg="(whole node)", comm=data.comm.comm_world, silent=True
    )
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
        "--zero_loaded_data",
        required=False,
        default=False,
        action="store_true",
        help="Zero out detector data loaded from disk",
    )

    # The operators and templates we want to configure from the command line
    # or a parameter file.

    operators = list()
    templates = list()

    # Loading data from disk is disabled by default
    so_wrk.setup_load_data_hdf5(operators)
    so_wrk.setup_load_data_books(operators)
    so_wrk.setup_load_data_context(operators)

    # Simulated observing is enabled by default
    so_wrk.setup_simulate_observing(parser, operators)

    so_wrk.setup_pointing(operators)
    so_wrk.setup_simple_noise_models(operators)
    so_wrk.setup_simulate_atmosphere_signal(operators)
    so_wrk.setup_simulate_sky_map_signal(operators)
    so_wrk.setup_simulate_conviqt_signal(operators)
    so_wrk.setup_simulate_detector_noise(operators)
    so_wrk.setup_simulate_calibration_error(operators)

    so_wrk.setup_save_data_hdf5(operators)

    so_wrk.setup_flag_noise_outliers(operators)
    so_wrk.setup_noise_estimation(operators)
    so_wrk.setup_raw_statistics(operators)
    so_wrk.setup_mapmaker(operators, templates)

    # setup mappraiser operator
    setup_mapmaker_mappraiser(parser, operators)

    job, config, otherargs, runargs = so_wrk.setup_job(
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

    data = simulate_data(job, otherargs, runargs, comm)

    if not job.operators.sim_atmosphere.cache_only:
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
