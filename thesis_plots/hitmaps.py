#!/usr/bin/env python3

from pathlib import Path

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

JZ = Path("../jz_out")
hits_large = hp.read_map(JZ / "opti" / "atm" / "ml" / "Hits_run0.fits", dtype=np.int32)
hits_small = hp.read_map(JZ / "leak" / "binned" / "ml" / "Hits_run0.fits", dtype=np.int32)

plt.figure(figsize=(8, 4))
hp.mollview(hits_large, sub=121, title="S1 hits", cmap="magma", xsize=5000)
hp.mollview(hits_small, sub=122, title="S2 hits", cmap="magma", xsize=5000)
plt.savefig("hitmaps.svg", bbox_inches="tight", dpi=600)
