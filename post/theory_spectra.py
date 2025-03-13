import camb
from camb import CAMBparams

# Planck FFP10 simulations
DEFAULT_PARS = camb.set_params(
    H0=67.01904,
    ombh2=0.02216571,
    omch2=0.1202944,
    omnuh2=6.451439e-4,
    tau=0.06018107,
    As=2.119631e-9,
    ns=0.9636852,
    # enough precision for lensing BB
    WantTensors=True,
    lmax=4000,
    lens_potential_accuracy=2,
)


def get_theory_powers(pars: CAMBparams = DEFAULT_PARS, lmax: int = 1_000):
    # calculate results for these parameters
    results = camb.get_results(pars)

    # get total CMB power spectra
    totCL = results.get_total_cls(lmax=lmax, CMB_unit="muK")
    return {
        "TT": totCL[:, 0],
        "EE": totCL[:, 1],
        "BB": totCL[:, 2],
        "TE": totCL[:, 3],
    }
