import marimo

__generated_with = "0.11.19"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import healpy as hp
    import jax
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pymaster as nmt

    from theory_spectra import get_theory_powers
    from utils import read_input_sky, read_mask
    return (
        Path,
        get_theory_powers,
        hp,
        jax,
        mpl,
        nmt,
        np,
        plt,
        read_input_sky,
        read_mask,
    )


@app.cell
def _(SCATTERS, full_cl, ls, mpl, plt, theory):
    _cl_hwp = full_cl["hwp"]
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4))
    _cmap = mpl.colormaps["Dark2"]
    _lines = []
    for _i, _scatter in enumerate(SCATTERS):
        _iqu = _cl_hwp[_scatter]["ml"]
        _pd = _cl_hwp[_scatter]["pd"]
        _axs[0].plot(_pd["ells"], _pd["cl_22"][0], color=_cmap(_i), label=str(_scatter))
        _axs[1].plot(_pd["ells"], _pd["cl_22"][3], color=_cmap(_i))
    _axs[0].plot(ls, theory["EE"], "k:", label="theory EE")
    _axs[1].plot(ls, theory["BB"], "k:", label="theory BB")
    _axs[0].legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Loading the data""")
    return


@app.cell
def _(Path):
    JZ = Path("..") / "jz_out"
    OPTI = JZ / "opti"
    SCATTERS = [0.001, 0.01, 0.1, 0.2]
    REF = "hits_10000"

    runs = {
        k_hwp: {
            scatter: {
                k_ml_pd: OPTI
                / ("var_increase_instr" + (f"_{k_hwp}" if k_hwp == "no_hwp" else ""))
                / f"scatter_{scatter}"
                / k_ml_pd
                for k_ml_pd in ["ml", "pd"]
            }
            for scatter in SCATTERS
        }
        for k_hwp in ["hwp", "no_hwp"]
    }
    return JZ, OPTI, REF, SCATTERS, runs


@app.cell
def _(np):
    def load_npz(path):
        data = np.load(path)
        # remove unnecessary dimensions
        return {k: data[k].squeeze() for k in data}
    return (load_npz,)


@app.cell
def _(np):
    def cl2dl(ell, cl):
        return ell * (ell + 1) / 2 / np.pi * cl
    return (cl2dl,)


@app.cell
def _(cl2dl, load_npz):
    def load_spectra(path, lmax=1000):
        cl = load_npz(path)
        l = cl.pop("ell_arr")
        good = l <= lmax
        return {"ells": l[good], **{k: cl2dl(l[good], cl[k][..., good]) for k in cl}}
    return (load_spectra,)


@app.cell
def _(JZ, REF, jax, load_spectra, runs):
    input_cl = load_spectra(JZ / "input_cells_mask_apo_1000.npz")
    full_cl = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"full_cl_{REF}.npz"), runs)
    noise_cl = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"noise_cl_{REF}.npz"), runs)
    return full_cl, input_cl, noise_cl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Comparing input sky with theory spectra""")
    return


@app.cell
def _(get_theory_powers, np):
    theory = get_theory_powers()
    ls = np.arange(theory["BB"].size)
    return ls, theory


@app.cell
def _(input_cl, ls, plt, theory):
    _l = input_cl["ells"]
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    _axs[0].plot(ls[2:], theory["EE"][2:], "k:", label="EE theory")
    _axs[0].plot(_l, input_cl["cl_22"][0], label="Input EE")
    _axs[1].plot(ls, theory["BB"], "k:", label="BB theory")
    _axs[1].plot(_l, input_cl["cl_22"][3], label="Input BB")
    _axs[2].plot(_l, input_cl["cl_22"][0] / theory["EE"][_l.astype(int)], label="EE ratio")
    _axs[2].plot(_l, input_cl["cl_22"][3] / theory["BB"][_l.astype(int)], label="BB ratio")
    _axs[2].set_ylim(0, 1.1)
    for _i, _ax in enumerate(_axs):
        _ax.set_xlabel(r"$\ell$")
        if _i < 2:
            _ax.set_ylabel(r"$C_\ell [\mu K]$")
        _ax.legend()
    plt.show()
    return


@app.cell
def _(JZ, read_input_sky, read_mask):
    sky = read_input_sky()
    mask_apo = read_mask(JZ / "mask_apo_1000.fits")
    return mask_apo, sky


@app.cell
def _(hp, mask_apo, np, plt, sky):
    plt.figure(figsize=(18, 8))
    hp.mollview(np.where(mask_apo > 0, sky[0] * mask_apo, np.nan), sub=131, title="input I")
    hp.mollview(np.where(mask_apo > 0, sky[1] * mask_apo, np.nan), sub=132, title="input Q")
    hp.mollview(np.where(mask_apo > 0, sky[2] * mask_apo, np.nan), sub=133, title="input U")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
