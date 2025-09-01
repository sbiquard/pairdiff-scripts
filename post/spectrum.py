import numpy as np
import pymaster as nmt
from timer import function_timer

NSIDE = 512
NPIX = 12 * NSIDE * NSIDE


@function_timer(thread="get_mask_apo")
def get_mask_apo(hits_map, hits_frac: float, aposize: float):
    # Define the mask by cutting above a hit threshold
    thresh = hits_frac * np.max(hits_map)
    mask = hits_map > thresh

    # Apodize the mask
    return nmt.mask_apodization(mask, aposize, apotype="C1")


def get_binning(delta_ell: int):
    # Initialize binning scheme
    return nmt.NmtBin.from_nside_linear(NSIDE, delta_ell)


def compute_spectra(m, mask_apo, binning):
    # Check input shape
    iqu = m.shape[0] == 3

    if iqu:
        # Initialize a spin-0 and spin-2 field
        f_0 = nmt.NmtField(mask_apo, [m[0]])
        f_2 = nmt.NmtField(mask_apo, m[1:])

        # Compute MASTER estimator
        # spin-0 x spin-0
        cl_00 = nmt.compute_full_master(f_0, f_0, binning)
        # spin-0 x spin-2
        cl_02 = nmt.compute_full_master(f_0, f_2, binning)
        # spin-2 x spin-2
        cl_22 = nmt.compute_full_master(f_2, f_2, binning)

        return {"cl_00": cl_00, "cl_02": cl_02, "cl_22": cl_22}
    else:
        # Assume we only have Q and U maps
        f_2 = nmt.NmtField(mask_apo, m)
        # spin-2 x spin-2
        cl_22 = nmt.compute_full_master(f_2, f_2, binning)

        return {"cl_22": cl_22}
