

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return


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
    hits = read_hits(LEAK / "correlated_no_hwp" / "ml")
    sky = {"IQU"[i]: arr for i, arr in enumerate(read_input_sky())}
    return hits, sky


@app.cell
def _(LEAK, hits, hp, np, read_maps, sky):
    correl_ml = {"maps": read_maps(LEAK / "correlated_no_hwp" / "ml"), "resid": {}}
    correl_pd = {"maps": read_maps(LEAK / "correlated_no_hwp" / "pd"), "resid": {}}
    miscal_ml = {"maps": read_maps(LEAK / "miscalibration_no_hwp" / "ml"), "resid": {}}
    miscal_pd = {"maps": read_maps(LEAK / "miscalibration_no_hwp" / "pd"), "resid": {}}

    # mask unseen pixels
    for run in (correl_ml, correl_pd, miscal_ml, miscal_pd):
        for k, v in run["maps"].items():
            run["maps"][k] = np.where(hits > 0, v, hp.UNSEEN)
            run["resid"][k] = np.where(hits > 0, v - sky[k], hp.UNSEEN)
    return correl_ml, correl_pd, miscal_ml, miscal_pd


@app.cell
def _(hp):
    from functools import partial

    gnomview = partial(hp.gnomview, rot=(25, -40), xsize=2500, cmap="bwr", min=-10, max=10, unit="µK")
    return (gnomview,)


@app.cell
def _(
    LEAK,
    correl_ml,
    correl_pd,
    gnomview,
    hits,
    hp,
    miscal_ml,
    miscal_pd,
    np,
    plt,
    sky,
):
    _fig = plt.figure(figsize=(17, 8))
    gnomview(np.where(hits > 0, sky["Q"], hp.UNSEEN), sub=[2, 5, 1], title="Input CMB")
    gnomview(correl_ml["maps"]["Q"], sub=[2, 5, 2], title="IQU good")
    gnomview(correl_pd["maps"]["Q"], sub=[2, 5, 3], title="PD good")
    gnomview(miscal_ml["maps"]["Q"], sub=[2, 5, 4], title="IQU miscal")
    gnomview(miscal_pd["maps"]["Q"], sub=[2, 5, 5], title="PD miscal")
    gnomview(np.where(hits > 0, sky["U"], hp.UNSEEN), sub=[2, 5, 6], title="")
    gnomview(correl_ml["maps"]["U"], sub=[2, 5, 7], title="")
    gnomview(correl_pd["maps"]["U"], sub=[2, 5, 8], title="")
    gnomview(miscal_ml["maps"]["U"], sub=[2, 5, 9], title="")
    gnomview(miscal_pd["maps"]["U"], sub=[2, 5, 10], title="")
    plt.savefig(LEAK.parent / "analysis" / "miscalibration.pdf", bbox_inches="tight")
    plt.gcf()
    return


@app.cell
def _(LEAK, correl_ml, correl_pd, gnomview, miscal_ml, miscal_pd, plt):
    _fig = plt.figure(figsize=(14, 8))
    gnomview(correl_ml["resid"]["Q"], sub=241, title="IQU good", min=-10, max=10)
    gnomview(correl_pd["resid"]["Q"], sub=242, title="PD good", min=-1e-1, max=1e-1)
    gnomview(miscal_ml["resid"]["Q"], sub=243, title="IQU miscal")
    gnomview(miscal_pd["resid"]["Q"], sub=244, title="PD miscal")
    gnomview(correl_ml["resid"]["U"], sub=245, title="", min=-10, max=10)
    gnomview(correl_pd["resid"]["U"], sub=246, title="", min=-1e-1, max=1e-1)
    gnomview(miscal_ml["resid"]["U"], sub=247, title="")
    gnomview(miscal_pd["resid"]["U"], sub=248, title="")
    plt.savefig(LEAK.parent / "analysis" / "miscalibration_resid.pdf", bbox_inches="tight")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
