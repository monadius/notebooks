# %%

import numpy as np
from matplotlib import pyplot as plt 
from lns.definitions import *

# %%

prec = 2 ** -8
delta = 2 ** -3
xs = np.arange(-2, prec, prec)
len(xs)

# %%
ys = phi_add(xs)
print(max_err(ys, fix_rnd(prec)(ys)), 0.5 * prec)
print(max_err(ys, fix_rnd_floor(prec)(ys)), prec)

# %%

exact = phi_add(xs)

ns = np.ceil(xs / delta) * delta
rs = ns - xs

approx = phi_add(ns) - rs * dphi_add(ns)

err = max_err(exact, approx)
err_bound = phi_add(-delta) - phi_add(0) + delta * dphi_add(0)
d = delta - prec
err_bound1 = phi_add(-d) - phi_add(0) + d * dphi_add(0)

print(err, err_bound, err / err_bound * 100)
print(err, err_bound1, err / err_bound1 * 100)


# %%

# plt.plot(xs[-500:], np.abs(approx - exact)[-500:])
plt.plot(xs, np.abs(approx - exact))
plt.plot(xs, [err_bound] * len(xs))

# %%

rnd = fix_rnd(prec)
exact_rnd = rnd(exact)
max_err(exact, exact_rnd), 0.5 * prec

# %%

rnd = fix_rnd_floor(prec)
exact_rnd = rnd(exact)
max_err(exact, exact_rnd), prec

# %%
rnd = fix_rnd_floor(prec)
print(max_err(taylor_add_rnd(rnd, delta, xs), exact))
ns = np.ceil(xs / delta) * delta
rs = ns - xs
rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))

print(max_err(rnd(rs * rnd(dphi_add(ns))), rs * dphi_add(ns)), prec)
print(max_err(rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns))), phi_add(ns) - rs * dphi_add(ns)), prec)

# %%

approx_rnd = rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))

err_rnd = max_err(exact, approx_rnd)
rnd_bound = (2 + delta) * (0.5 * eps)
a1 = max_err(phi_add(ns), rnd(phi_add(ns)))
a2 = max_err(dphi_add(ns), rnd(dphi_add(ns)))
rnd_bound = a1 + delta * a2 + 0.5 * eps

err_bound_rnd = err_bound + rnd_bound
err_bound_rnd1 = err_bound1 + rnd_bound

print(err_rnd, err_bound_rnd, err_rnd / err_bound_rnd * 100)
print(err_rnd, err_bound_rnd1, err_rnd / err_bound_rnd1 * 100)
print(a1, a2, 0.5 * eps)

# %%

def get_add_error(prec, delta):
    xs = np.arange(-3, prec, prec)
    exact = phi_add(xs)
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    rnd = fix_rnd(prec)
    approx_rnd = rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))
    rnd_bound = (2 + delta) * (0.5 * prec)
    a1 = max_err(phi_add(ns), rnd(phi_add(ns)))
    a2 = max_err(dphi_add(ns), rnd(dphi_add(ns)))
    rnd_bound1 = a1 + delta * a2 + 0.5 * prec
    err_bound = phi_add(-delta) - phi_add(0) + delta * dphi_add(0)
    err_rnd = max_err(exact, approx_rnd)
    d = delta - prec
    err_bound1 = phi_add(-d) - phi_add(0) + d * dphi_add(0)
    err_bound_rnd = err_bound + rnd_bound
    err_bound_rnd1 = err_bound1 + rnd_bound1
    return err_rnd / err_bound_rnd * 100, err_rnd / err_bound_rnd1 * 100

def get_sub_error(prec, delta):
    xs = np.arange(-4, -1 + prec, prec)
    exact = phi_sub(xs)
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    rnd = fix_rnd(prec)
    approx_rnd = rnd(phi_sub(ns)) - rnd(rs * rnd(dphi_sub(ns)))
    rnd_bound = (2 + delta) * (0.5 * prec)

    a1 = max_err(phi_sub(ns), rnd(phi_sub(ns)))
    a2 = max_err(dphi_sub(ns), rnd(dphi_sub(ns)))
    rnd_bound1 = a1 + delta * a2 + 0.5 * prec

    err_bound = -phi_sub(-1 - delta) + phi_sub(-1) - delta * dphi_sub(-1)
    d = delta - prec
    err_bound1 = -phi_sub(-1 - d) + phi_sub(-1) - d * dphi_sub(-1)
    err_rnd = max_err(exact, approx_rnd)
    err_bound_rnd = err_bound + rnd_bound
    err_bound_rnd1 = err_bound1 + rnd_bound1
    return err_rnd / err_bound_rnd * 100, err_rnd / err_bound_rnd1 * 100


# %%

add_cases = [
    (2**-8, 2**-3),
    (2**-8, 2**-4),
    (2**-8, 2**-5),
    (2**-16, 2**-4),
    (2**-16, 2**-6),
    (2**-16, 2**-8),
    # (2**-23, 2**-4),
    # (2**-23, 2**-6),
    # (2**-23, 2**-8),
]

res = [get_add_error(*case) for case in add_cases]
plt.bar([f'{int(np.log2(p))}, {int(np.log2(d))}' for p, d in add_cases], [x for x, y  in res])
# %%

sub_cases = [
    (2**-8, 2**-3),
    (2**-8, 2**-4),
    (2**-8, 2**-5),
    (2**-16, 2**-4),
    (2**-16, 2**-6),
    (2**-16, 2**-8),
    # (2**-23, 2**-4),
    # (2**-23, 2**-6),
    # (2**-23, 2**-8),
]

res = [get_sub_error(*case) for case in sub_cases]
plt.bar([f'{int(np.log2(p))}, {int(np.log2(d))}' for p, d in sub_cases], [x for x, y in res])
print(res)

# %%
