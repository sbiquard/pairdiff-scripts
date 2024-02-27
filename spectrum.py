import numpy as np
import pymaster as nmt

from utils import timer

NSIDE = 512
NPIX = 12 * NSIDE * NSIDE


@timer
def get_mask_apo_and_binning(hits_map, min_hits=1000):
    # Apodize the mask on a scale of ~10deg
    mask = nmt.mask_apodization(hits_map > min_hits, 10.0, apotype="C1")

    # Initialize binning scheme with 10 ells per bandpower
    b = nmt.NmtBin.from_nside_linear(NSIDE, 10)

    return mask, b


def compute_spectra(m, mask_apo, binning):
    m = np.array(m)
    iqu = m.size == 3

    if iqu:
        # Initialize a spin-0 and spin-2 field
        f_0 = nmt.NmtField(mask_apo, [m[:, 0]])
        f_2 = nmt.NmtField(mask_apo, [m[:, 1], m[:, 2]])

        # Compute MASTER estimator
        # spin-0 x spin-0
        cl_00 = nmt.compute_full_master(f_0, f_0, binning)
        # spin-0 x spin-2
        cl_02 = nmt.compute_full_master(f_0, f_2, binning)
        # spin-2 x spin-2
        cl_22 = nmt.compute_full_master(f_2, f_2, binning)

        return cl_00, cl_02, cl_22
    else:
        # Assume we only have Q and U maps
        f_2 = nmt.NmtField(mask_apo, [m[:, 1], m[:, 2]])
        # spin-2 x spin-2
        cl_22 = nmt.compute_full_master(f_2, f_2, binning)

        return cl_22
