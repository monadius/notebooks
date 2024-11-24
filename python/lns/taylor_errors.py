# Run with python3 -m lns.taylor_errors from the parent directory

# %%

import numpy as np
from matplotlib import pyplot as plt
from lns.definitions import *

# %%

def get_add_error(prec: float, delta: float) -> tuple[float, float, float]:
    xs = np.arange(-3, prec, prec)
    rnd, eps = fix_rnd(prec), 0.5 * prec
    # Exact values computed with float64
    exact = phi_add(xs)
    # Approximate values
    approx_rnd = taylor_add_rnd(rnd, delta, xs)
    # The actual error
    err_rnd = max_err(exact, approx_rnd)
    # The error bound
    err_bound_rnd = taylor_add_rnd_err_bound(prec, delta)
    # Attempt to compute a more accurate bound
    ns = np.ceil(xs / delta) * delta
    a1 = max_err(phi_add(ns), rnd(phi_add(ns)))
    a2 = max_err(dphi_add(ns), rnd(dphi_add(ns)))
    rnd_bound1 = a1 + delta * a2 + eps
    d = delta - prec
    err_bound1 = phi_add(-d) - phi_add(0) + d * dphi_add(0)
    err_bound_rnd1 = err_bound1 + rnd_bound1
    return err_rnd, err_bound_rnd, err_bound_rnd1

def get_sub_error(prec: float, delta: float) -> tuple[float, float, float]:
    xs = np.arange(-4, -1 + prec, prec)
    rnd, eps = fix_rnd(prec), 0.5 * prec
    # Exact values computed with float64
    exact = phi_sub(xs)
    # Approximate values
    approx_rnd = taylor_sub_rnd(rnd, delta, xs)
    # The actual error
    err_rnd = max_err(exact, approx_rnd)
    # The error bound
    err_bound_rnd = taylor_sub_rnd_err_bound(prec, delta)
    # Attempt to compute a more accurate bound
    ns = np.ceil(xs / delta) * delta
    a1 = max_err(phi_sub(ns), rnd(phi_sub(ns)))
    a2 = max_err(dphi_sub(ns), rnd(dphi_sub(ns)))
    rnd_bound1 = a1 + delta * a2 + eps
    d = delta - prec
    err_bound1 = -phi_sub(-1 - d) + phi_sub(-1) - d * dphi_sub(-1)
    err_bound_rnd1 = err_bound1 + rnd_bound1
    return err_rnd, err_bound_rnd, err_bound_rnd1


# %%

# Test cases (p, d) with prec = 2**p, delta = 2**d
test_cases: list[tuple[int, int]] = [
    (-8, -3),
    (-8, -4),
    (-8, -5),
    (-16, -4),
    (-16, -6),
    (-16, -8),
    # (-23, -4),
    # (-23, -6),
    # (-23, -8),
]

xs = [str(case) for case in test_cases]
res_add = [get_add_error(2 ** p, 2 ** d) for p, d in test_cases]
plot = plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_add])
plt.show()

# %%
res_sub = [get_sub_error(2 ** p, 2 ** d) for p, d in test_cases]
plot = plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_sub])
plt.show()

# %% Plot improved and standard errors together
plt.bar(xs, [err / bound2 for err, bound1, bound2 in res_add])
plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_add])

# %%
plt.bar(xs, [err / bound2 for err, bound1, bound2 in res_sub])
plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_sub])

# %%

def plot_error(prec: int):
    deltas = [*range(prec // 2 - 1, -2)]
    errs = [get_add_error(2 ** prec, 2 ** d) for d in deltas]
    fig = plt.figure(figsize = (16, 9))
    plot = fig.add_subplot()
    plot.plot(deltas, np.log2([err[0] for err in errs]), color='red', linewidth = 3)
    plot.plot(deltas, np.log2([err[1] for err in errs]), color='green', linewidth = 3)
    plot.set_xlabel('log2 Δ')
    plot.set_ylabel('log2 err')
    plot.legend(['actual', 'bound'])
    plot.grid(which='both', axis='both', linestyle='-.')
    plt.suptitle(f'Fixed point precision: 2 ** {prec}', fontsize=16)
    fig.show()
    plt.savefig(f'taylor_add_{abs(prec)}.png')
# %%
plot_error(-10)
plot_error(-15)
# plot_error(-20)
# plot_error(-24)

# %%
