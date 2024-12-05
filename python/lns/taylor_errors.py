# Run with python3 -m lns.taylor_errors from the parent directory

# %%

import numpy as np
from matplotlib import pyplot as plt
from definitions import *

# %%

def get_add_error(prec: float, delta: float, rounding_mode: RoundingMode) -> tuple[float, float, float]:
    """Computes actual and theoretical error bounds for Taylor approximation of phi_add.
    
    Evaluates the Taylor approximation of phi_add with fixed-point rounding over a range
    of inputs and compares against exact values to determine actual error. Also computes
    theoretical error bounds.

    Args:
        prec: The precision (step size) for fixed-point arithmetic
        delta: The step size for the Taylor approximation
        rounding_mode: The fixed-point arithmetic rounding mode

    Returns:
        A tuple containing:
        - The actual maximum error observed
        - A theoretical error bound from taylor_add_rnd_err_bound
        - A tighter error bound computed using more detailed analysis
    """
    xs = np.arange(-3, prec, prec)
    rnd = fix_rnd(prec, rounding_mode)
    nearest = rounding_mode == RoundingMode.NEAREST
    # Set the eps value
    if nearest:
        eps = 0.5 * prec
    else:
        eps = prec
    # Compute the fixed-point rounding error bound
    match rounding_mode:
        case RoundingMode.NEAREST | RoundingMode.FAITHFUL:
            rnd_bound = (2 + delta) * eps
        case RoundingMode.FLOOR | RoundingMode.CEIL:
            rnd_bound = (1 + delta) * eps
        case _: raise ValueError(f"Rounding mode {rounding_mode} not supported")
    # Exact values computed with float64
    exact = phi_add(xs)
    # Approximate values
    approx_rnd = taylor_add_rnd(rnd, delta, xs)
    # The actual error
    err_rnd = max_err(exact, approx_rnd)
    # The error bound
    err_bound_rnd = taylor_add_err_bound(delta) + rnd_bound
    # Attempt to compute a more accurate bound
    ns = np.ceil(xs / delta) * delta
    a1 = max_err(phi_add(ns), rnd(phi_add(ns)))
    a2 = max_err(dphi_add(ns), rnd(dphi_add(ns)))
    # TODO: not correct for FAITHFUL
    rnd_bound1 = a1 + delta * a2 + eps if nearest else eps + delta * a2
    d = delta - prec
    err_bound1 = phi_add(-d) - phi_add(0) + d * dphi_add(0)
    err_bound_rnd1 = err_bound1 + rnd_bound1
    return err_rnd, err_bound_rnd, err_bound_rnd1

def get_sub_error(prec: float, delta: float, rounding_mode: RoundingMode) -> tuple[float, float, float]:
    xs = np.arange(-4, -1 + prec, prec)
    rnd = fix_rnd(prec, rounding_mode)
    nearest = rounding_mode == RoundingMode.NEAREST
    # Set the eps value
    if nearest:
        eps = 0.5 * prec
    else:
        eps = prec
    match rounding_mode:
        case RoundingMode.NEAREST | RoundingMode.FAITHFUL:
            rnd_bound = (2 + delta) * eps
        case RoundingMode.FLOOR | RoundingMode.CEIL:
            rnd_bound = (1 + delta) * eps
        case _: raise ValueError(f"Rounding mode {rounding_mode} not supported")
    # Exact values computed with float64
    exact = phi_sub(xs)
    # Approximate values
    approx_rnd = taylor_sub_rnd(rnd, delta, xs)
    # The actual error
    err_rnd = max_err(exact, approx_rnd)
    # The error bound
    err_bound_rnd = taylor_sub_err_bound(delta) + rnd_bound
    # Attempt to compute a more accurate bound
    ns = np.ceil(xs / delta) * delta
    a1 = max_err(phi_sub(ns), rnd(phi_sub(ns)))
    a2 = max_err(dphi_sub(ns), rnd(dphi_sub(ns)))
    # TODO: not correct for FAITHFUL
    rnd_bound1 = a1 + delta * a2 + eps if nearest else eps + delta * a2
    d = delta - prec
    err_bound1 = -phi_sub(-1 - d) + phi_sub(-1) - d * dphi_sub(-1)
    err_bound_rnd1 = err_bound1 + rnd_bound1
    return err_rnd, err_bound_rnd, err_bound_rnd1

def get_add_average_error(prec: float, delta: float, nearest: bool = True) -> tuple[float, float]:
    """Computes actual average error for Taylor approximation of phi_add."""
    xs = np.arange(-3, prec, prec)
    rnd = fix_rnd(prec, RoundingMode.NEAREST) if nearest else fix_rnd_floor(prec)
    exact = phi_add(xs)
    # Approximate values
    approx = taylor_add_rnd(rnd, delta, xs)
    return avg_abs_err(exact, approx), avg_err(exact, approx)


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
res_add = [get_add_error(2 ** p, 2 ** d, RoundingMode.NEAREST) for p, d in test_cases]
plot = plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_add])
plt.xlabel('(log2 prec, log2 Δ)')
plt.ylabel('err / err bound')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.title('Taylor Addition (rounding to nearest)')
plt.savefig('taylor_add_err_nearest.png')
plt.show()

res_add = [get_add_error(2 ** p, 2 ** d, RoundingMode.FLOOR) for p, d in test_cases]
plot = plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_add])
plt.xlabel('(log2 prec, log2 Δ)')
plt.ylabel('err / err bound')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.title('Taylor Addition (directed rounding)')
plt.savefig('taylor_add_err_directed.png')
plt.show()


# %%
res_sub = [get_sub_error(2 ** p, 2 ** d, RoundingMode.NEAREST) for p, d in test_cases]
plot = plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_sub])
plt.xlabel('(log2 prec, log2 Δ)')
plt.ylabel('err / err bound')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.title('Taylor Subtraction (rounding to nearest)')
plt.savefig('taylor_sub_err_nearest.png')
plt.show()

res_sub = [get_sub_error(2 ** p, 2 ** d, RoundingMode.FLOOR) for p, d in test_cases]
plot = plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_sub])
plt.xlabel('(log2 prec, log2 Δ)')
plt.ylabel('err / err bound')
plt.gca().set_axisbelow(True)
plt.grid(axis='y')
plt.title('Taylor Subtraction (directed rounding)')
plt.savefig('taylor_sub_err_directed.png')
plt.show()


# %% Plot improved and standard errors together
plt.bar(xs, [err / bound2 for err, bound1, bound2 in res_add])
plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_add])

# %%
plt.bar(xs, [err / bound2 for err, bound1, bound2 in res_sub])
plt.bar(xs, [err / bound1 for err, bound1, bound2 in res_sub])

# %%

def plot_error(prec: int, name: str, rounding_mode: RoundingMode):
    f = {'add': get_add_error, 'sub': get_sub_error, 'add average': get_add_average_error}[name]
    deltas = [*range(prec // 2 - 1, -2)]
    errs = [f(2 ** prec, 2 ** d, rounding_mode) for d in deltas]
    fig = plt.figure(figsize = (16, 9))
    plot = fig.add_subplot()
    plot.plot(deltas, np.log2([err[0] for err in errs]), color='red', linewidth=3)
    plot.plot(deltas, np.log2([err[1] for err in errs]), color='green', linewidth=3)
    # plot.plot(deltas, np.log2([err[2] for err in errs]), color='brown', linewidth=3)
    plot.set_xlabel('log2 Δ')
    plot.set_ylabel('log2 err')
    plot.legend(['avg of abs', 'avg'] if 'average' in name else ['actual', 'bound'])
    plot.grid(which='both', axis='both', linestyle='-.')
    plt.suptitle(f'{name.capitalize()}: fixed point precision = 2 ** {prec}, {rounding_mode}', fontsize=16)
    fig.show()
    # plt.savefig(f'taylor_{name}_{abs(prec)}_{"nearest" if nearest else "directed"}.png')

# %%
plot_error(-10, 'add', RoundingMode.FLOOR)
plot_error(-15, 'add', RoundingMode.FLOOR)
# plot_error(-23, 'add', RoundingMode.FLOOR)
# plot_error(-20, 'add', RoundingMode.FLOOR)
# plot_error(-24, 'add', RoundingMode.FLOOR)

# %%

plot_error(-10, 'add', RoundingMode.FAITHFUL)
plot_error(-15, 'add', RoundingMode.FAITHFUL)


# %%
plot_error(-10, 'add', RoundingMode.NEAREST)
plot_error(-15, 'add', RoundingMode.NEAREST)
# plot_error(-23, 'add', RoundingMode.NEAREST)


# %%
plot_error(-10, 'sub', RoundingMode.FLOOR)
plot_error(-15, 'sub', RoundingMode.FLOOR)
# plot_error(-23, 'add', RoundingMode.FLOOR)
# plot_error(-20, 'add', RoundingMode.FLOOR)
# plot_error(-24, 'add', RoundingMode.FLOOR)

# %%
plot_error(-10, 'sub', RoundingMode.NEAREST)
plot_error(-15, 'sub', RoundingMode.NEAREST)
# plot_error(-23, 'add', RoundingMode.NEAREST)

# %%
plot_error(-10, 'add average', RoundingMode.NEAREST)
plot_error(-15, 'add average', RoundingMode.NEAREST)

# %%
plot_error(-10, 'add average', RoundingMode.FLOOR)
plot_error(-15, 'add average', RoundingMode.FLOOR)

# %%
