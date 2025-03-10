import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import Array, PRNGKeyArray

MIN_SCATTER = 1e-3
MAX_SCATTER = 0.2
NUM_SCATTER = 100

SAMPLES = 10_000
TRUNC = 0.9


def _rel_diff_sqr(a, b):
    return (a**2 - b**2) / (a**2 + b**2)


def get_scatters(
    scatter_bounds: tuple[float, float] = (MIN_SCATTER, MAX_SCATTER),
    scatter_num: int = NUM_SCATTER,
):
    min_scatter, max_scatter = scatter_bounds
    return jnp.geomspace(min_scatter, max_scatter, scatter_num)


def get_epsilon_samples(key: PRNGKeyArray, scatters: Array):
    # sample scatter values
    scale = scatters[None, :, None]
    rngdata = 1 + scale * jr.truncated_normal(
        key,
        -TRUNC / scale,  # lower bound
        TRUNC / scale,  # upper bound
        shape=(2, *scatters.shape, SAMPLES),
    )

    para, perp = rngdata
    return _rel_diff_sqr(para, perp)


def main():
    # Sample epsilon values
    scatters = get_scatters()
    key = jr.key(1234)
    eps = get_epsilon_samples(key, scatters)
    alpha = 1 / (1 - eps**2)

    # Plot the expected variance increase
    fig, ax = plt.subplots()
    ax.set(
        xlabel="Scatter around nominal NET [%]",
        ylabel="Variance increase [%]",
    )

    x = scatters * 100
    expect = jnp.mean(alpha, axis=-1)
    y = (expect - 1) * 100
    q = jnp.array([90, 99])
    p = (jnp.percentile(alpha, q, axis=-1) - 1) * 100

    ax.loglog(x, y, "k", label="expected increase")

    for q_, p_, ls_ in zip(q, p, ["--", ":"]):
        ax.loglog(x, p_, "k" + ls_, label=f"{q_}th percentile")

    # Fit a line to the expected increase
    log_x = jnp.log(x)
    p = jnp.polyfit(log_x, jnp.log(y), 1)
    y_fit = jnp.exp(jnp.polyval(p, log_x))
    ax.loglog(x, y_fit, linewidth=0.8, label=f"fit {p[0]:.2f} x + {p[1]:.2f}")

    ax.legend()
    plt.show()
    # fig.savefig("../jz_out/analysis/expected_variance_increase.png")


if __name__ == "__main__":
    main()
