import marimo

__generated_with = "0.12.7"
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
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt
    import numpy as np
    import pymaster as nmt
    import seaborn as sns
    from matplotlib.collections import LineCollection

    from theory_spectra import get_theory_powers
    from utils import read_input_sky, read_mask
    return (
        LineCollection,
        Path,
        get_theory_powers,
        hp,
        jax,
        mpl,
        mtick,
        nmt,
        np,
        plt,
        read_input_sky,
        read_mask,
        sns,
    )


@app.cell
def _(mo):
    mo.md(r"""### Relative increase -- white noise""")
    return


@app.cell(hide_code=True)
def _(JZ, SCATTERS, mpl, mtick, plt, sns, stack_noise_cl_white):
    _fig = plt.figure(layout="constrained", figsize=(12, 10))
    _fig.suptitle("Relative noise power increase in pair differencing maps with white noise")

    _subfigs = _fig.subfigures(2, 1)
    _subfigs[0].suptitle("HWP on")
    _subfigs[1].suptitle("HWP off")

    _cmap = sns.color_palette("flare", as_cmap=True)
    _colors = [_cmap(i / (len(SCATTERS) - 1)) for i in range(len(SCATTERS))]

    for _khwp, _subfig in zip(["hwp", "no_hwp"], _subfigs):
        _cl = stack_noise_cl_white[_khwp]
        _axs = _subfig.subplots(1, 2, sharex=True, sharey=True)
        _axs[0].set_title("EE")
        _axs[1].set_title("BB")

        # Set appropriate axis limits
        for _ax in _axs:
            _ax.set_xlim(2, 1000)
            _ax.grid(True)
            _ax.set_xlabel("Multipole $\ell$")
            _ax.set_ylabel("Relative power increase")
            _ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            _ax.label_outer()

        for _is, _scatter in enumerate(SCATTERS):
            _iqu = _cl[_scatter]["ml"]
            _pd = _cl[_scatter]["pd"]
            _ells = _iqu["ells"][0]  # identical for all realizations
            _diff = _pd["cl_22"] / _iqu["cl_22"] - 1
            for _ax, _idx in zip(_axs, (0, 3)):
                # _line_data = [np.column_stack([_ells[2:], _diff.mean(axis=0)[_idx][2:]]) for _scatter in SCATTERS]
                # _lines = LineCollection(_line_data, colors=_colors)
                # _ax.add_collection(_lines)
                _y = _diff.mean(axis=0)[_idx]
                _yerr = _diff.std(axis=0)[_idx]
                _ax.errorbar(_ells[1:], _y[1:], yerr=_yerr[1:], fmt=".", color=_colors[_is], linewidth=0.5)

                # print increase averaged over bins
                print(_khwp, _scatter, "EE" if _idx == 0 else "BB", f"{_y.mean():.2%}")

        # Add colorbar with scatter values
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=mpl.colors.LogNorm(vmin=min(SCATTERS), vmax=max(SCATTERS)))
        _cbar = _fig.colorbar(_sm, ax=_axs, pad=0.01)
        _cbar.set_ticks(SCATTERS)
        _cbar.set_ticklabels([f"{s:.1%}" for s in SCATTERS])
        _cbar.set_label("Scatter")

        # Autoscale axes after adding collections
        _axs[0].autoscale()
        _axs[1].autoscale()

        # _axs[1].plot(ls[2:], 3 * prim_BB[2:], "k:", label="primordial BB (r=0.03)")
        # _axs[1].legend()

        # _axs[0].set_ylim(-1e-6, 1e-4)
        # _axs[1].set_ylim(top=3e-6)

        for _ax in _axs:
            _ax.set_xlim(right=600)

    _fig.savefig(JZ / "analysis" / "optimality" / "var_increase_spectra_white_relative", dpi=200)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Absolute increase -- white noise""")
    return


@app.cell(hide_code=True)
def _(JZ, SCATTERS, mpl, plt, sns, stack_noise_cl_white):
    _fig = plt.figure(layout="constrained", figsize=(12, 10))
    _fig.suptitle("Absolute noise power increase in pair differencing maps with white noise")

    _subfigs = _fig.subfigures(2, 1)
    _subfigs[0].suptitle("HWP on")
    _subfigs[1].suptitle("HWP off")

    _cmap = sns.color_palette("flare", as_cmap=True)
    _colors = [_cmap(i / (len(SCATTERS) - 1)) for i in range(len(SCATTERS))]

    for _khwp, _subfig in zip(["hwp", "no_hwp"], _subfigs):
        _cl = stack_noise_cl_white[_khwp]
        _axs = _subfig.subplots(1, 2, sharex=True, sharey=True)
        _axs[0].set_title("EE")
        _axs[1].set_title("BB")

        # Set appropriate axis limits
        for _ax in _axs:
            _ax.set_xlim(2, 1000)
            _ax.grid(True)
            _ax.set_xlabel("Multipole $\ell$")
            _ax.set_ylabel("Power $[\mu K^2]$")
            # _ax.set_xscale("asinh", linear_width=10)
            _ax.label_outer()

        for _is, _scatter in enumerate(SCATTERS):
            _iqu = _cl[_scatter]["ml"]
            _pd = _cl[_scatter]["pd"]
            _ells = _iqu["ells"][0]  # identical for all realizations
            _diff = _pd["cl_22"] - _iqu["cl_22"]
            for _ax, _idx in zip(_axs, (0, 3)):
                _y = _diff.mean(axis=0)[_idx]
                _yerr = _diff.std(axis=0)[_idx]
                _ax.errorbar(_ells[2:], _y[2:], yerr=_yerr[2:], fmt=".", color=_colors[_is], linewidth=0.5)

        # Add colorbar with scatter values
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=mpl.colors.LogNorm(vmin=min(SCATTERS), vmax=max(SCATTERS)))
        _cbar = _fig.colorbar(_sm, ax=_axs, pad=0.01)
        _cbar.set_ticks(SCATTERS)
        _cbar.set_ticklabels([f"{s:.1%}" for s in SCATTERS])
        _cbar.set_label("Scatter")

        # Autoscale axes after adding collections
        _axs[0].autoscale()
        _axs[1].autoscale()

        for _ax in _axs:
            _ax.set_xlim(right=600)

    _fig.savefig(JZ / "analysis" / "optimality" / "var_increase_spectra_white_absolute", dpi=200)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Relative increase -- instrumental noise""")
    return


@app.cell(hide_code=True)
def _(JZ, SCATTERS, mpl, mtick, plt, sns, stack_noise_cl_instr):
    _fig = plt.figure(layout="constrained", figsize=(12, 10))
    _fig.suptitle("Relative noise power increase in pair differencing maps with 1/f noise")

    _subfigs = _fig.subfigures(2, 1)
    _subfigs[0].suptitle("HWP on")
    _subfigs[1].suptitle("HWP off")

    _cmap = sns.color_palette("flare", as_cmap=True)
    _colors = [_cmap(i / (len(SCATTERS) - 1)) for i in range(len(SCATTERS))]

    for _khwp, _subfig in zip(["hwp", "no_hwp"], _subfigs):
        _cl = stack_noise_cl_instr[_khwp]
        _axs = _subfig.subplots(1, 2, sharex=True, sharey=True)
        _axs[0].set_title("EE")
        _axs[1].set_title("BB")

        # Set appropriate axis limits
        for _ax in _axs:
            _ax.set_xlim(2, 1000)
            _ax.grid(True)
            _ax.set_xlabel("Multipole $\ell$")
            _ax.set_ylabel("Relative power increase")
            _ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            _ax.label_outer()

        for _is, _scatter in enumerate(SCATTERS):
            _iqu = _cl[_scatter]["ml"]
            _pd = _cl[_scatter]["pd"]
            _ells = _iqu["ells"][0]  # identical for all realizations
            _diff = _pd["cl_22"] / _iqu["cl_22"] - 1
            for _ax, _idx in zip(_axs, (0, 3)):
                # _line_data = [np.column_stack([_ells[2:], _diff.mean(axis=0)[_idx][2:]]) for _scatter in SCATTERS]
                # _lines = LineCollection(_line_data, colors=_colors)
                # _ax.add_collection(_lines)
                _y = _diff.mean(axis=0)[_idx]
                _yerr = _diff.std(axis=0)[_idx]
                _ax.errorbar(_ells[2:], _y[2:], yerr=_yerr[2:], fmt=".", color=_colors[_is], linewidth=0.5)

                # print increase in white noise domain
                print(_khwp, _scatter, "EE" if _idx == 0 else "BB", f"{_y.mean():.2%}")

        # Add colorbar with scatter values
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=mpl.colors.LogNorm(vmin=min(SCATTERS), vmax=max(SCATTERS)))
        _cbar = _fig.colorbar(_sm, ax=_axs, pad=0.01)
        _cbar.set_ticks(SCATTERS)
        _cbar.set_ticklabels([f"{s:.1%}" for s in SCATTERS])
        _cbar.set_label("Scatter")

        # Autoscale axes after adding collections
        _axs[0].autoscale()
        _axs[1].autoscale()

        if _khwp == "hwp":
            _ax.set_ylim(top=0.2)
        else:
            _ax.set_ylim(-0.1, 0.4)

        # _axs[1].plot(ls[2:], 3 * prim_BB[2:], "k:", label="primordial BB (r=0.03)")
        # _axs[1].legend()

        # _axs[0].set_ylim(-1e-6, 1e-4)
        # _axs[1].set_ylim(top=3e-6)

        for _ax in _axs:
            _ax.set_xlim(right=600)

    _fig.savefig(JZ / "analysis" / "optimality" / "var_increase_spectra_instr_relative", dpi=200)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Absolute increase -- instrumental noise""")
    return


@app.cell(hide_code=True)
def _(JZ, SCATTERS, mpl, plt, sns, stack_noise_cl_instr):
    _fig = plt.figure(layout="constrained", figsize=(12, 10))
    _fig.suptitle("Absolute noise power increase in pair differencing maps with 1/f noise")

    _subfigs = _fig.subfigures(2, 1)
    _subfigs[0].suptitle("HWP on")
    _subfigs[1].suptitle("HWP off")

    _cmap = sns.color_palette("flare", as_cmap=True)
    _colors = [_cmap(i / (len(SCATTERS) - 1)) for i in range(len(SCATTERS))]

    for _khwp, _subfig in zip(["hwp", "no_hwp"], _subfigs):
        _cl = stack_noise_cl_instr[_khwp]
        _axs = _subfig.subplots(1, 2, sharex=True, sharey=True)
        _axs[0].set_title("EE")
        _axs[1].set_title("BB")

        # Set appropriate axis limits
        for _ax in _axs:
            _ax.set_xlim(2, 1000)
            _ax.grid(True)
            _ax.set_xlabel("Multipole $\ell$")
            _ax.set_ylabel("Power $[\mu K^2]$")
            # _ax.set_xscale("asinh", linear_width=10)
            _ax.label_outer()

        for _is, _scatter in enumerate(SCATTERS):
            _iqu = _cl[_scatter]["ml"]
            _pd = _cl[_scatter]["pd"]
            _ells = _iqu["ells"][0]  # identical for all realizations
            _diff = _pd["cl_22"] - _iqu["cl_22"]
            for _ax, _idx in zip(_axs, (0, 3)):
                # _line_data = [np.column_stack([_ells[2:], _diff.mean(axis=0)[_idx][2:]]) for _scatter in SCATTERS]
                # _lines = LineCollection(_line_data, colors=_colors)
                # _ax.add_collection(_lines)
                _y = _diff.mean(axis=0)[_idx]
                _yerr = _diff.std(axis=0)[_idx]
                _ax.errorbar(_ells[2:], _y[2:], yerr=_yerr[2:], fmt=".", color=_colors[_is], linewidth=0.5)
                # _ax.plot(_ells, _iqu["cl_22"][_idx].mean(axis=0), marker='.', color=_colors[_is])
                # _ax.plot(_ells, _pd["cl_22"][_idx].mean(axis=0), marker='d', color=_colors[_is])

        # Add colorbar with scatter values
        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=mpl.colors.LogNorm(vmin=min(SCATTERS), vmax=max(SCATTERS)))
        _cbar = _fig.colorbar(_sm, ax=_axs, pad=0.01)
        _cbar.set_ticks(SCATTERS)
        _cbar.set_ticklabels([f"{s:.1%}" for s in SCATTERS])
        _cbar.set_label("Scatter")

        # Autoscale axes after adding collections
        _axs[0].autoscale()
        _axs[1].autoscale()

        if _khwp == "no_hwp":
            _ax.set_ylim(-5e-7, 8e-6)

        # _axs[1].plot(ls[2:], 3 * prim_BB[2:], "k:", label="primordial BB (r=0.03)")
        # _axs[1].legend()

        # _axs[0].set_ylim(-1e-6, 1e-4)
        # _axs[1].set_ylim(top=3e-6)

        for _ax in _axs:
            _ax.set_xlim(right=600)

    _fig.savefig(JZ / "analysis" / "optimality" / "var_increase_spectra_instr_absolute", dpi=200)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Loading the data""")
    return


@app.cell(hide_code=True)
def _(Path, __file__):
    JZ = Path(__file__).parents[1].absolute() / "jz_out"
    OPTI = JZ / "opti"
    SCATTERS = [0.001, 0.01, 0.1, 0.2]
    # SCATTERS = [0.001, 0.01]
    # MASK_REF = "hits_10000"
    MASK_REF = "hits_10000_apo_20"

    runs = {
        k_white: {
            k_hwp: {
                scatter: {
                    k_ml_pd: [
                        OPTI
                        / (
                            "var_increase"
                            + ("_instr" if k_white != "white" else "")
                            + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")
                        )
                        / f"{real + 1:03d}"
                        / f"scatter_{scatter}"
                        # / (k_ml_pd if k_ml_pd == "ml" else "pd_new")
                        / k_ml_pd
                        for real in range(25)
                    ]
                    for k_ml_pd in ["ml", "pd"]
                }
                for scatter in SCATTERS
            }
            for k_hwp in ["hwp", "no_hwp"]
        }
        for k_white in ["white", "instr"]
    }
    return JZ, MASK_REF, OPTI, SCATTERS, runs


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
def _(Path, cl2dl, load_npz):
    def load_spectra(path: Path | list[Path], lmax=1000, dl=False):
        cl = load_npz(path)
        l = cl.pop("ell_arr")
        good = l <= lmax
        if dl:
            return {"ells": l[good], **{k: cl2dl(l[good], cl[k][..., good]) for k in cl}}
        else:
            return {"ells": l[good], **{k: cl[k][..., good] for k in cl}}
    return (load_spectra,)


@app.cell
def _(JZ, MASK_REF, jax, load_spectra, runs):
    input_cl = load_spectra(JZ / "input_cells_mask_apo_1000.npz")
    input_dl = load_spectra(JZ / "input_cells_mask_apo_1000.npz", dl=True)
    full_cl_white = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz"), runs["white"])
    full_cl_instr = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz"), runs["instr"])
    full_dl_white = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz", dl=True), runs["white"])
    full_dl_instr = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"full_cl_{MASK_REF}.npz", dl=True), runs["instr"])
    noise_cl_white = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz"), runs["white"])
    noise_cl_instr = jax.tree.map(lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz"), runs["instr"])
    noise_dl_white = jax.tree.map(
        lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz", dl=True), runs["white"]
    )
    noise_dl_instr = jax.tree.map(
        lambda x: load_spectra(x / "spectra" / f"noise_cl_{MASK_REF}.npz", dl=True), runs["instr"]
    )
    return (
        full_cl_instr,
        full_cl_white,
        full_dl_instr,
        full_dl_white,
        input_cl,
        input_dl,
        noise_cl_instr,
        noise_cl_white,
        noise_dl_instr,
        noise_dl_white,
    )


@app.cell
def _(
    full_cl_instr,
    full_cl_white,
    full_dl_instr,
    full_dl_white,
    jax,
    noise_cl_instr,
    noise_cl_white,
    noise_dl_instr,
    noise_dl_white,
    np,
):
    def _stack(cl):
        # assuming homogeneous shapes
        return {k: np.stack(tuple(cl[i][k] for i in range(len(cl)))) for k in cl[0]}


    stack_full_cl_white = jax.tree.map(_stack, full_cl_white, is_leaf=lambda x: isinstance(x, list))
    stack_full_cl_instr = jax.tree.map(_stack, full_cl_instr, is_leaf=lambda x: isinstance(x, list))
    stack_full_dl_white = jax.tree.map(_stack, full_dl_white, is_leaf=lambda x: isinstance(x, list))
    stack_full_dl_instr = jax.tree.map(_stack, full_dl_instr, is_leaf=lambda x: isinstance(x, list))
    stack_noise_cl_white = jax.tree.map(_stack, noise_cl_white, is_leaf=lambda x: isinstance(x, list))
    stack_noise_cl_instr = jax.tree.map(_stack, noise_cl_instr, is_leaf=lambda x: isinstance(x, list))
    stack_noise_dl_white = jax.tree.map(_stack, noise_dl_white, is_leaf=lambda x: isinstance(x, list))
    stack_noise_dl_instr = jax.tree.map(_stack, noise_dl_instr, is_leaf=lambda x: isinstance(x, list))
    return (
        stack_full_cl_instr,
        stack_full_cl_white,
        stack_full_dl_instr,
        stack_full_dl_white,
        stack_noise_cl_instr,
        stack_noise_cl_white,
        stack_noise_dl_instr,
        stack_noise_dl_white,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Comparing input sky with theory spectra""")
    return


@app.cell
def _(cl2dl, get_theory_powers, jax, np):
    theory_cl = get_theory_powers()
    ls = np.arange(theory_cl["BB"].size)
    theory_dl = jax.tree.map(lambda x: cl2dl(ls, x), theory_cl)
    prim_BB = theory_cl["BB"] - get_theory_powers(r=0)["BB"]
    prim_BB_dl = cl2dl(ls, prim_BB)
    return ls, prim_BB, prim_BB_dl, theory_cl, theory_dl


@app.cell
def _(input_dl, ls, plt, theory_dl):
    _l = input_dl["ells"]
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    _axs[0].plot(ls[2:], theory_dl["EE"][2:], "k:", label="EE theory")
    _axs[0].plot(_l, input_dl["cl_22"][0], label="Input EE")
    _axs[1].plot(ls, theory_dl["BB"], "k:", label="BB theory")
    _axs[1].plot(_l, input_dl["cl_22"][3], label="Input BB")
    _axs[2].plot(_l, input_dl["cl_22"][0] / theory_dl["EE"][_l.astype(int)], label="EE ratio")
    _axs[2].plot(_l, input_dl["cl_22"][3] / theory_dl["BB"][_l.astype(int)], label="BB ratio")
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
