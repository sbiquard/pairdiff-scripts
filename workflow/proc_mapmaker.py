"""Utility to use the Mappraiser operator in a TOAST workflow"""

import numpy as np
import pymappraiser.toast as ptoast
import toast
import toast.ops
from astropy import units as u
from pymappraiser.toast import Mappraiser
from sotodlib.toast import ops as so_ops
from sotodlib.toast.workflows.job import workflow_timer
from toast.observation import default_values as defaults


def setup_mapmaker(parser, operators):
    """Add commandline args and operators for the MAPPRAISER mapmaker.

    Args:
        parser (ArgumentParser):  The parser to update.
        operators (list):  The list of operators to extend.

    Returns:
        None

    """
    parser.add_argument(
        "--ref",
        required=False,
        default="run0",
        help="Reference that is added to the name of the output maps.",
    )

    if ptoast.mappraiser.available():
        operators.append(Mappraiser(name="mappraiser", enabled=True))


@workflow_timer
def mapmaker(job, otherargs, runargs, data):
    """Run the MAPPRAISER mapmaker.

    Args:
        job (namespace):  The configured operators and templates for this job.
        otherargs (namespace):  Other commandline arguments.
        runargs (namespace):  Job related runtime parameters.
        data (Data):  The data container.

    Returns:
        None

    """
    # Configured operators for this job
    job_ops = job.operators

    if ptoast.mappraiser.available() and job_ops.mappraiser.enabled:
        job_ops.mappraiser.params["path_output"] = otherargs.out_dir
        job_ops.mappraiser.params["ref"] = otherargs.ref
        job_ops.mappraiser.pixel_pointing = job.pixels_final
        job_ops.mappraiser.stokes_weights = job.weights_final
        job_ops.mappraiser.apply(data)
