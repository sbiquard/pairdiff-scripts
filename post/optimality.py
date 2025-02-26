#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import toml
from furax import TreeOperator
from furax.obs.stokes import Stokes, StokesIQU, StokesQU

from .furax_preconditioner import BJPreconditioner
from .timer import Timer
from .utils import get_last_ref

OPTI = Path("..") / "out" / "opti"
SAVE_PLOTS_DIR = Path("..") / "out" / "analysis" / "optimality"
SAVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def my_savefig(fig, title: str, close: bool = True):
    fig.savefig(SAVE_PLOTS_DIR / title, bbox_inches="tight")
    if close:
        plt.close(fig)


def read_hits(run: Path):
    ref = get_last_ref(run)
    return jnp.array(hp.fitsfunc.read_map(run / f"Hits_{ref}.fits", field=None, dtype=jnp.int32))


runs_white = {
    k_ml_or_pd: {
        k_hwp: {
            "none": OPTI
            / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else ""))
            / "no_scatter"
            / k_ml_or_pd,
            "same": OPTI
            / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else ""))
            / "same_scatter"
            / k_ml_or_pd,
            "opposite": OPTI
            / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else ""))
            / "opposite_scatter"
            / k_ml_or_pd,
            "random": OPTI
            / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else ""))
            / "random_scatter"
            / k_ml_or_pd,
        }
        for k_hwp in ["hwp", "no_hwp"]
    }
    for k_ml_or_pd in ["ml", "pd"]
}

with Timer(thread="read-hits"):
    hitmaps = {k: read_hits(v["hwp"]["none"]) for k, v in runs_white.items()}

# THRESH = 1_000
THRESH = 10_000

with Timer("create-masks"):
    MASKS = {k: hitmaps[k] * (2 if k == "pd" else 1) > THRESH for k in hitmaps}


def mask_outside(maps_, fill_value=jnp.nan):
    ml_or_pd = "ml" if isinstance(maps_, StokesIQU) else "pd"
    mask = MASKS[ml_or_pd]
    return jax.tree.map(lambda leaf: jnp.where(mask, leaf, fill_value), maps_)


def read_maps(run: Path):
    # ref of the run
    ref = get_last_ref(run)

    # read logged param file
    params = toml.load(run / "config_log.toml")
    params = params["operators"]["mappraiser"]

    # do we have iqu maps or just qu?
    stokes = "IQU" if not params["pair_diff"] or params["estimate_spin_zero"] else "QU"

    mapQ = 1e6 * jnp.array(hp.fitsfunc.read_map(run / f"mapQ_{ref}.fits", field=None))
    mapU = 1e6 * jnp.array(hp.fitsfunc.read_map(run / f"mapU_{ref}.fits", field=None))

    if "I" in stokes:
        mapI = 1e6 * jnp.array(hp.fitsfunc.read_map(run / f"mapI_{ref}.fits", field=None))
        stokes_maps = StokesIQU(mapI, mapQ, mapU)
    else:
        stokes_maps = StokesQU(mapQ, mapU)

    return mask_outside(stokes_maps)


def read_cond(run: Path):
    ref = get_last_ref(run)
    cond = jnp.array(hp.fitsfunc.read_map(run / f"Cond_{ref}.fits", field=None))
    return mask_outside(cond)


def read_epsilon(run: Path):
    return np.load(run / "epsilon_dist.npy")


def read_prec(run: Path, stokes: str | None = None):
    # ref of the run
    ref = get_last_ref(run)

    # read logged param file
    params = toml.load(run / "config_log.toml")
    params = params["operators"]["mappraiser"]

    # do we have iqu maps or just qu?
    if stokes is None:
        stokes = "IQU" if not params["pair_diff"] or params["estimate_spin_zero"] else "QU"
    klass = Stokes.class_for(stokes)

    precQQ = 1e12 * jnp.array(hp.fitsfunc.read_map(run / f"precQQ_{ref}.fits", field=None))
    precQU = 1e12 * jnp.array(hp.fitsfunc.read_map(run / f"precQU_{ref}.fits", field=None))
    precUU = 1e12 * jnp.array(hp.fitsfunc.read_map(run / f"precUU_{ref}.fits", field=None))
    shape = (precQQ.size,)

    if "I" in stokes:
        precII = 1e12 * jnp.array(hp.fitsfunc.read_map(run / f"precII_{ref}.fits", field=None))
        precIQ = 1e12 * jnp.array(hp.fitsfunc.read_map(run / f"precIQ_{ref}.fits", field=None))
        precIU = 1e12 * jnp.array(hp.fitsfunc.read_map(run / f"precIU_{ref}.fits", field=None))
        tree = StokesIQU(
            StokesIQU(precII, precIQ, precIU),
            StokesIQU(precIQ, precQQ, precQU),
            StokesIQU(precIU, precQU, precUU),
        )
    else:
        tree = StokesQU(
            StokesQU(precQQ, precQU),
            StokesQU(precQU, precUU),
        )

    masked_tree = mask_outside(tree)
    return BJPreconditioner(masked_tree, in_structure=klass.structure_for(shape, jnp.float32))


def read_input_sky(iqu=True):
    filename = Path.cwd().parent / "ffp10_lensed_scl_100_nside0512.fits"
    if iqu:
        # read all fields
        sky = hp.fitsfunc.read_map(filename, field=None)
        sky_in = StokesIQU(
            i=jnp.array(sky[0]),
            q=jnp.array(sky[1]),
            u=jnp.array(sky[2]),
        )
    else:
        # read only relevant fields
        sky = hp.fitsfunc.read_map(filename, field=[1, 2])
        sky_in = StokesQU(
            q=jnp.array(sky[0]),
            u=jnp.array(sky[1]),
        )
    return mask_outside(1e6 * sky_in)


with Timer(thread="read-maps"):
    maps_white = {
        k: {kk: {k3: read_maps(v3) for k3, v3 in vv.items()} for kk, vv in v.items()}
        for k, v in runs_white.items()
    }

with Timer(thread="read-precs"):
    precs_white = {
        k: {kk: {k3: read_prec(v3) for k3, v3 in vv.items()} for kk, vv in v.items()}
        for k, v in runs_white.items()
    }

with Timer(thread="read-precs-ideal-qu"):
    precs_ideal_qu = {
        k: {kk: read_prec(vv, stokes="QU") for kk, vv in v.items()}
        for k, v in runs_white["ml"].items()
    }

with Timer(thread="read-epsilon"):
    epsilons_white = {
        k: {kk: {k3: read_epsilon(v3) for k3, v3 in vv.items()} for kk, vv in v.items()}
        for k, v in runs_white.items()
    }

with Timer(thread="scale-precs-by-hits"):
    precs_scaled_by_hits = {
        k: jax.tree.map(lambda leaf: leaf * hitmaps[k], v) for k, v in precs_white.items()
    }

with Timer(thread="read-sky"):
    input_sky = read_input_sky()

with Timer(thread="compute-residuals"):
    residuals_white = jax.tree.map(
        lambda x: x - type(x).from_iquv(input_sky.i, input_sky.q, input_sky.u, None),
        maps_white,
        is_leaf=lambda x: isinstance(x, Stokes),
    )


LONRA = [-95, 135]
LATRA = [-70, -10]
CMAP = "bwr"
# ROT = [15, -40]

my_cartview = partial(hp.cartview, lonra=LONRA, latra=LATRA, cmap=CMAP)
my_mollview = partial(hp.mollview, cmap=CMAP)


def get_figsize_for(stokes: str, proj: str):
    n = len(stokes)
    if proj == "cart":
        return (4 * n, 2 * n)
    if proj == "moll":
        return (4 * n, 4 * n)
    msg = f"{proj!r} not supported"
    raise NotImplementedError(msg)


def plot_stokes_tree_operator(
    op,
    proj: str = "cart",
    title: str | None = None,
):
    leaves, _ = jax.tree.flatten(op.tree)
    stokes = op.tree.stokes
    ns = len(stokes)

    plot_func = partial(
        my_cartview if proj == "cart" else my_mollview,
        unit="$\\mu K_{CMB}^2$",
        cmap=CMAP,
        # return_projected_map=True,
    )

    # hf, ha = plt.subplots(ns, ns, figsize=get_figsize_for(stokes, proj))
    f = plt.figure(figsize=get_figsize_for(stokes, proj))
    if title is not None:
        f.suptitle(title)

    for i, stoke_in in enumerate(stokes):
        for j, stoke_out in enumerate(stokes):
            # for j, stoke_out in enumerate(stokes[i:]):
            # ax = ha[i, j]
            if j < i:
                # Only plot upper triangle
                # ax.set_axis_off()
                continue

            # Index in the flat list of leaves
            n = ns * i + j

            # Use healpy plotting function to get the projected map, and close the figure right away

            # amp = get_amp(leaves[n]) if stoke_out != stoke_in else None
            # min_ = -amp if amp is not None else None
            # max_ = amp or None
            plot_func(leaves[n], sub=[ns, ns, n + 1])
            # plt.close()

            # # Re-plot the projected map on our figure
            # pos = plot_projected_map(pj, ax=ax, title=f"{stoke_in}{stoke_out}")

            # # Draw the color bar
            # hf.colorbar(pos, ax=ax, location='bottom', shrink=0.6, pad=0.05, label="$\\mu K_{CMB}^2$")
    return f


title_helper_ml_pd = {
    "ml": "full IQU",
    "pd": "pair diff",
}

title_helper_run = {
    "none": "no scatter",
    "same": "same scatter",
    "opposite": "opposite scatter",
    "random": "random scatter",
}

with Timer(thread="plot-cov-matrices"):
    for k_ml_pd, val_ml_pd in precs_white.items():
        for k_hwp, val_hwp in val_ml_pd.items():
            for k_run, val_run in val_hwp.items():
                helper_ml_pd = title_helper_ml_pd[k_ml_pd]
                helper_run = title_helper_run[k_run]
                # Noise covariance
                fig = plot_stokes_tree_operator(
                    val_run,
                    title=f"Covariance matrix ({helper_ml_pd}, {k_hwp}, {helper_run})",
                )
                my_savefig(fig, f"noise_cov_{k_ml_pd}_{k_hwp}_{helper_run.replace(' ', '_')}")
                # Noise covariance scaled by hits
                fig = plot_stokes_tree_operator(
                    precs_scaled_by_hits[k_ml_pd][k_hwp][k_run],
                    title=f"Cov scaled by hits ({helper_ml_pd}, {k_hwp}, {helper_run})",
                )
                my_savefig(
                    fig, f"noise_cov_scaled_{k_ml_pd}_{k_hwp}_{helper_run.replace(' ', '_')}"
                )


# Compute ratios of covariance
with Timer(thread="compute-ratio-over-ideal"):
    pd_over_ideal = jax.tree.map(
        lambda pd, ideal: (pd @ ideal.I).reduce(),
        precs_white["pd"],
        precs_ideal_qu,
        is_leaf=lambda x: isinstance(x, TreeOperator),
    )

with Timer(thread="plot-variance-increase"):
    for k_hwp, val_hwp in pd_over_ideal.items():
        for k_run, val_run in val_hwp.items():
            fig, ax = plt.subplots()
            qq = val_run.tree.q.q
            uu = val_run.tree.u.u
            ax.hist(qq[~jnp.isnan(qq)], bins="auto", histtype="step", label="QQ", density=False)
            ax.hist(uu[~jnp.isnan(uu)], bins="auto", histtype="step", label="UU", density=False)
            dist = epsilons_white["pd"][k_hwp][k_run]
            ax.axvline(
                (1 / (1 - dist**2)).mean(),
                color="k",
                ls="--",
                label=r"$\langle 1 / (1 - \epsilon^2) \rangle$",
            )
            ax.legend()
            run_title = title_helper_run[k_run]
            hwp_title = k_hwp.replace(" ", "_")
            ax.set(
                xlabel="Variance increase in pixel",
                ylabel="Number of pixels",
                title=f"Histogram of variance increase ({hwp_title}, {run_title})",
            )
            my_savefig(fig, title=f"variance_increase_{k_hwp}_{run_title.replace(' ', '_')}")
