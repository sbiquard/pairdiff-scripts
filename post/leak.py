import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import healpy as hp
    import matplotlib.pyplot as plt
    import numpy as np
    return Path, hp, np, plt


@app.cell
def _(Path, __file__):
    LEAK = Path(__file__).parents[1] / "jz_out" / "leak"
    return (LEAK,)


@app.cell
def _():
    from utils import read_maps, read_input_sky, read_hits
    return read_hits, read_input_sky, read_maps


@app.cell
def _(LEAK, read_hits, read_input_sky):
    hits = read_hits(LEAK / "correlated" / "ml")
    sky = {"IQU"[i]: arr for i, arr in enumerate(read_input_sky())}
    return hits, sky


@app.cell
def _(LEAK, hits, hp, np, read_maps, sky):
    perfect_ml = {"maps": read_maps(LEAK / "correlated" / "ml"), "resid": {}}
    perfect_pd = {"maps": read_maps(LEAK / "correlated" / "pd"), "resid": {}}
    miscal_ml = {"maps": read_maps(LEAK / "miscalibration" / "ml"), "resid": {}}
    miscal_pd = {"maps": read_maps(LEAK / "miscalibration" / "pd"), "resid": {}}

    # mask unseen pixels
    for run in (perfect_ml, perfect_pd, miscal_ml, miscal_pd):
        for k, v in run["maps"].items():
            run["maps"][k] = np.where(hits > 0, v, hp.UNSEEN)
            run["resid"][k] = np.where(hits > 0, v - sky[k], hp.UNSEEN)
    return k, miscal_ml, miscal_pd, perfect_ml, perfect_pd, run, v


@app.cell
def _(hp):
    from functools import partial

    gnomview = partial(hp.gnomview, rot=(25, -40), xsize=3000, cmap="bwr", min=-10, max=10, unit="µK")
    return gnomview, partial


@app.cell
def _(LEAK, gnomview, miscal_ml, miscal_pd, perfect_ml, plt):
    _fig = plt.figure(figsize=(11, 8))
    gnomview(perfect_ml["maps"]["Q"], sub=231, title="No miscal")
    gnomview(miscal_ml["maps"]["Q"], sub=232, title="IQU miscal")
    gnomview(miscal_pd["maps"]["Q"], sub=233, title="PD miscal")
    gnomview(perfect_ml["maps"]["U"], sub=234, title="")
    gnomview(miscal_ml["maps"]["U"], sub=235, title="")
    gnomview(miscal_pd["maps"]["U"], sub=236, title="")
    plt.savefig(LEAK.parent / "analysis" / "miscalibration.pdf", bbox_inches="tight")
    plt.gcf()
    return


@app.cell
def _(LEAK, gnomview, miscal_ml, miscal_pd, perfect_ml, perfect_pd, plt):
    _fig = plt.figure(figsize=(14, 8))
    gnomview(perfect_ml["resid"]["Q"], sub=241, title="IQU good", min=-1e-1, max=1e-1)
    gnomview(perfect_pd["resid"]["Q"], sub=242, title="PD good", min=-1e-6, max=1e-6)
    gnomview(miscal_ml["resid"]["Q"], sub=243, title="IQU miscal")
    gnomview(miscal_pd["resid"]["Q"], sub=244, title="PD miscal")
    gnomview(perfect_ml["resid"]["U"], sub=245, title="", min=-1e-1, max=1e-1)
    gnomview(perfect_pd["resid"]["U"], sub=246, title="", min=-1e-6, max=1e-6)
    gnomview(miscal_ml["resid"]["U"], sub=247, title="")
    gnomview(miscal_pd["resid"]["U"], sub=248, title="")
    plt.savefig(LEAK.parent / "analysis" / "miscalibration_resid.pdf", bbox_inches="tight")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
