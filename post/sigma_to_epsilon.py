import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

# from matplotlib import ticker


def rel_diff_sqr(a, b):
    return (a**2 - b**2) / (a**2 + b**2)


MIN_SCATTER = 1e-3
MAX_SCATTER = 0.2
DISCRETIZATION = 100

SAMPLES = 10_000
TRUNC = 0.9

# sample scatter values
scatters = jnp.geomspace(MIN_SCATTER, MAX_SCATTER, DISCRETIZATION)
scale = scatters[None, :, None]
rngdata = 1 + scale * jr.truncated_normal(
    jr.key(123),
    -TRUNC / scale,  # lower bound
    TRUNC / scale,  # upper bound
    shape=(2, *scatters.shape, SAMPLES),
)
# print(jnp.min(rngdata, axis=(0, 2)))
# print(jnp.max(rngdata, axis=(0, 2)))

para, perp = rngdata
# print(para.shape, perp.shape)
epsilon = rel_diff_sqr(para, perp)
alpha = 1 / (1 - epsilon**2)

# Plot everything
fig, ax = plt.subplots()
ax.set(
    xlabel="Scatter around nominal NET [%]",
    ylabel="variance increase [%]",
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
