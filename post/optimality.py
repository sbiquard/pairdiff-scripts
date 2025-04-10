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
from furax_preconditioner import BJPreconditioner
from sigma_to_epsilon import get_epsilon_samples, get_scatters
from timer import Timer
from utils import get_last_ref

OPTI = Path("..") / "out" / "opti"
SAVE_PLOTS_DIR = Path("..") / "out" / "analysis" / "optimality"
SAVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCATTERS = [0.001, 0.01, 0.1, 0.2]  # 0.3 is crap


def my_savefig(fig, title: str, close: bool = True, dpi=200):
    fig.savefig(SAVE_PLOTS_DIR / title, bbox_inches="tight", dpi=dpi)
    if close:
        plt.close(fig)


runs = {
    "white": {
        k_ml_or_pd: {
            k_hwp: {
                "none": OPTI / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")) / "no_scatter" / k_ml_or_pd,
                "same": OPTI / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")) / "same_scatter" / k_ml_or_pd,
                "opposite": OPTI / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")) / "opposite_scatter" / k_ml_or_pd,
                "random": OPTI / ("white" + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")) / "random_scatter" / k_ml_or_pd,
            }
            for k_hwp in ["hwp", "no_hwp"]
        }
        for k_ml_or_pd in ["ml", "pd"]
    },
    "var_increase": {
        k_ml_or_pd: {
            k_hwp: {
                scatter: OPTI / ("var_increase" + (f"_{k_hwp}" if k_hwp == "no_hwp" else "")) / f"scatter_{scatter}" / k_ml_or_pd
                for scatter in SCATTERS
            }
            for k_hwp in ["hwp", "no_hwp"]
        }
        for k_ml_or_pd in ["ml", "pd"]
    }
}  # fmt: skip


def read_hits(run: Path):
    ref = get_last_ref(run)
    return jnp.array(hp.fitsfunc.read_map(run / f"Hits_{ref}.fits", field=None, dtype=jnp.int32))


with Timer(thread="read-hits"):
    hitmaps = {k: read_hits(runs["white"][k]["hwp"]["none"]) for k in ["ml", "pd"]}

# THRESH = {"ml": 1_000, "pd": 500}
THRESH = 10_000
MASK = hitmaps["ml"] > THRESH


def mask_outside(maps_, fill_value=jnp.nan):
    return jax.tree.map(lambda leaf: jnp.where(MASK, leaf, fill_value), maps_)


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
    return BJPreconditioner(masked_tree, in_structure=klass.structure_for(shape))


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
    maps = jax.tree.map(read_maps, runs)

with Timer(thread="read-precs"):
    precs = jax.tree.map(read_prec, runs)

with Timer(thread="read-precs-ideal-qu"):
    precs_ideal_qu = jax.tree.map(
        partial(read_prec, stokes="QU"),
        {k: v["ml"] for k, v in runs.items()},
    )

with Timer(thread="read-epsilon"):
    epsilons = jax.tree.map(read_epsilon, runs)

with Timer(thread="scale-precs-by-hits"):
    precs_scaled = {
        k: {kk: jax.tree.map(lambda leaf: leaf * hitmaps[kk], vv) for kk, vv in v.items()}
        for k, v in precs.items()
    }

with Timer(thread="read-sky"):
    input_sky = read_input_sky()

with Timer(thread="compute-residuals"):
    residuals = jax.tree.map(
        lambda x: x - type(x).from_iquv(input_sky.i, input_sky.q, input_sky.u, None),
        maps,
        is_leaf=lambda x: isinstance(x, Stokes),
    )


LONRA = [-95, 135]
LATRA = [-70, -10]
CMAP = "bwr"

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
    )

    f = plt.figure(figsize=get_figsize_for(stokes, proj))
    if title is not None:
        f.suptitle(title)

    for i, stoke_in in enumerate(stokes):
        for j, stoke_out in enumerate(stokes):
            if j < i:
                # Only plot upper triangle
                continue

            # Index in the flat list of leaves
            n = ns * i + j
            plot_func(leaves[n], sub=[ns, ns, n + 1])
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
    for k_ml_pd, val_ml_pd in precs["white"].items():
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
                    precs_scaled["white"][k_ml_pd][k_hwp][k_run],
                    title=f"Cov scaled by hits ({helper_ml_pd}, {k_hwp}, {helper_run})",
                )
                my_savefig(
                    fig, f"noise_cov_scaled_{k_ml_pd}_{k_hwp}_{helper_run.replace(' ', '_')}"
                )


# Compute ratios of covariance
with Timer(thread="compute-pd-over-ideal"):
    pd_over_ideal = jax.tree.map(
        lambda pd, ideal: (pd @ ideal.I).reduce(),
        {k: v["pd"] for k, v in precs.items()},
        precs_ideal_qu,
        is_leaf=lambda x: isinstance(x, TreeOperator),
    )

with Timer(thread="plot-variance-increase-white"):
    for k_hwp, val_hwp in pd_over_ideal["white"].items():
        for k_run, val_run in val_hwp.items():
            fig, ax = plt.subplots()
            qq = val_run.tree.q.q
            uu = val_run.tree.u.u
            ax.hist(qq[~jnp.isnan(qq)], bins="auto", histtype="step", label="QQ", density=False)
            ax.hist(uu[~jnp.isnan(uu)], bins="auto", histtype="step", label="UU", density=False)
            dist = epsilons["white"]["pd"][k_hwp][k_run]
            ax.axvline(
                (1 / (1 - dist**2)).mean(),
                color="k",
                ls="--",
                label=r"$\langle 1 / (1 - \epsilon^2) \rangle$",
            )
            ax.legend()
            run_title = title_helper_run[k_run]
            hwp_title = k_hwp.replace("_", " ")
            ax.set(
                xlabel="Variance increase in pixel",
                ylabel="Number of pixels",
                title=f"Histogram of variance increase ({hwp_title}, {run_title})",
            )
            my_savefig(fig, title=f"variance_increase_{k_hwp}_{run_title.replace(' ', '_')}")

with Timer(thread="plot-variance-increase-scatter"):
    # plotting expected variance increase as a function of scatter

    # sample scatter values
    scatters = get_scatters((SCATTERS[0], 1.1 * SCATTERS[-1]))
    eps = get_epsilon_samples(jax.random.key(1426), scatters)
    alpha = 1 / (1 - eps**2)

    # loop over hwp and no_hwp cases
    for k_hwp, val in pd_over_ideal["var_increase"].items():
        fig, ax = plt.subplots()
        hwp_title = k_hwp.replace("_", " ")
        ax.set(
            xlabel="Scatter around nominal NET [%]",
            ylabel="Variance increase [%]",
            title=f"Variance increase per pixel ({hwp_title})",
        )

        # Expected variance increase as a function of scatter
        x = scatters * 100
        y = (alpha.mean(axis=-1) - 1) * 100
        ax.loglog(x, y, "k", label="expected increase")

        q = jnp.array([90, 99])
        p_qq = []
        p_uu = []
        means_qq = []
        means_uu = []

        for scatter in SCATTERS:
            alpha_qq = (val[scatter].tree.q.q - 1) * 100
            alpha_uu = (val[scatter].tree.u.u - 1) * 100

            # scatter_pct = scatter * 100

            # parts_qq = ax.violinplot(
            #     alpha_qq[~jnp.isnan(alpha_qq)],
            #     positions=[scatter_pct],
            #     widths=5,
            #     showextrema=False,
            #     showmeans=False,
            #     showmedians=False,
            #     side="low",
            # )

            # parts_uu = ax.violinplot(
            #     alpha_uu[~jnp.isnan(alpha_uu)],
            #     positions=[scatter_pct],
            #     widths=5,
            #     showextrema=False,
            #     showmeans=False,
            #     showmedians=False,
            #     side="high",
            # )

            # for pc in parts_qq["bodies"]:
            #     pc.set_facecolor("orange")

            # for pc in parts_uu["bodies"]:
            #     pc.set_facecolor("green")

            means_qq.append(jnp.nanmean(alpha_qq))
            means_uu.append(jnp.nanmean(alpha_uu))

            p_qq.append(jnp.nanpercentile(alpha_qq, q))
            p_uu.append(jnp.nanpercentile(alpha_uu, q))

        # ax.scatter(np.array(SCATTERS) * 100, means_qq, marker=5, color="orange", label="QQ average")
        # ax.scatter(np.array(SCATTERS) * 100, means_uu, marker=4, color="green", label="UU average")

        x_m = jnp.array(SCATTERS) * 100

        # interpolate measured percentiles in log space
        y_90 = jnp.interp(jnp.log(x), jnp.log(x_m), jnp.log(jnp.array([_p[0] for _p in p_qq])))
        y_99 = jnp.interp(jnp.log(x), jnp.log(x_m), jnp.log(jnp.array([_p[1] for _p in p_qq])))

        ax.fill_between(x, y, jnp.exp(y_90), alpha=0.5, color="g", label="90th percentile")
        ax.fill_between(x, y, jnp.exp(y_99), alpha=0.5, color="orange", label="99th percentile")

        # for i, q_ in enumerate(q):
        #     ax.plot(x_m, [_p[i] for _p in p_qq], color="orange", label=f"QQ {q_}th percentile")
        #     ax.plot(x_m, [_p[i] for _p in p_qq], color="green", label=f"UU {q_}th percentile")

        ax.scatter(x_m, means_qq, marker=5, color="r", label="QQ average")
        ax.scatter(x_m, means_uu, marker=4, color="r", label="UU average")

        ax.legend()
        ax.grid(True)

        my_savefig(fig, f"variance_increase_scatter_{k_hwp}")
