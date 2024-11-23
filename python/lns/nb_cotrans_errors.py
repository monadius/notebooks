# %%

import numpy as np
from matplotlib import pyplot as plt 

def max_err(exact, approx):
    return np.max(np.abs(exact - approx))

# %%

def fix_rnd(prec): 
    return lambda xs: np.round(xs * (1 / prec)) * prec

def phi_sub(x):
    return np.log2(1 - 2 ** x)

def dphi_sub(x):
    return 2 ** x / (2 ** x - 1)

def taylor_sub(delta, xs):
    if np.any(xs > -1):
        raise ValueError('taylor_sub: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_sub(ns) - rs * dphi_sub(ns)

def taylor_sub_rnd(prec, delta, xs):
    # print(max(xs))
    if np.any(xs > -1):
        raise ValueError('taylor_sub_rnd: xs > -1')
    rnd = fix_rnd(prec)
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_sub(ns)) - rnd(rs * rnd(dphi_sub(ns)))

def taylor_sub_err(delta):
    return -phi_sub(-1 - delta) + phi_sub(-1) - delta * dphi_sub(-1)

def taylor_sub_rnd_err(prec, delta):
    eps = 0.5 * prec
    return taylor_sub_err(delta) + (2 + delta) * eps

def ind(delta, xs):
    return (np.ceil(xs / delta) - 1) * delta

def rem(delta, xs):
    return ind(delta, xs) - xs

def k(delta, xs):
    return xs - phi_sub(ind(delta, xs)) + phi_sub(rem(delta, xs))

def k_rnd(prec, delta, xs):
    rnd = fix_rnd(prec)
    return xs - rnd(phi_sub(ind(delta, xs))) + rnd(phi_sub(rem(delta, xs)))

def cotrans2(delta, da, xs):
    return phi_sub(ind(da, xs)) + taylor_sub(delta, k(da, xs))

def cotrans2_rnd(prec, delta, da, xs):
    rnd = fix_rnd(prec)
    return rnd(phi_sub(ind(da, xs))) + taylor_sub_rnd(prec, delta, k_rnd(prec, da, xs))

def cotrans3(delta, da, db, xs):
    return phi_sub(ind(db, xs)) + taylor_sub(delta, k(db, xs))

def cotrans3_rnd(prec, delta, da, db, xs):
    rnd = fix_rnd(prec)
    rab = rem(db, xs)
    res = np.zeros(len(xs))
    special = rab >= -da
    incl = rab < -da
    rab, xs, ys = rab[incl], xs[incl], xs[special]
    rb = ind(da, rab)
    k1 = k_rnd(prec, da, rab)
    k2 = xs + rnd(phi_sub(rb)) + taylor_sub_rnd(prec, delta, k1) - rnd(phi_sub(ind(db, xs)))
    res[incl] = rnd(phi_sub(ind(db, xs))) + taylor_sub_rnd(prec, delta, k2)
    res[special] = cotrans2_rnd(prec, delta, db, ys)
    return res

# %%
prec = 2 ** -24
eps = 0.5 * prec

da = 2 ** -20
db = 2 ** -10
delta = 2 ** -3

print(da >= 4 * eps, db >= 8 * eps + delta ** 2 * np.log(2))
xs = np.arange(-1, -db, prec)
ks = k(db, xs)
ks[ks >= -1]


# %%

prec = 2**-8
da = 2**-6
db = 2**-3
xs = np.arange(-1, -db, prec)
ys = xs[rem(db, xs) >= -da]
k(da, rem(db, xs)) >= -1

# %%

prec = 2**-8
eps = 0.5 * prec
da = 2**-6
db = 2**-3
delta = 2**-3

xs = np.arange(-db, -da, prec)
exact = phi_sub(xs)
approx = cotrans2_rnd(prec, delta, da, xs)
err = max_err(exact, approx)
err_bound = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
err, err_bound, err / err_bound * 100

# %%

def cotrans2_error(prec, da, db, delta):
    eps = 0.5 * prec
    xs = np.arange(-db, -da, prec)
    exact = phi_sub(xs)
    approx = cotrans2_rnd(prec, delta, da, xs)
    err = max_err(exact, approx)
    err_bound = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
    return err / err_bound * 100

cotrans2_cases = [
    # p, da, db, delta
    (-8, -6, -3, -3),
    (-8, -6, -3, -4),
    (-8, -5, -2, -3),
    (-8, -5, -2, -4),
    (-16, -12, -6, -4),
    (-16, -12, -6, -6),
    (-16, -10, -5, -4),
    (-16, -10, -5, -6),
]

errs = [cotrans2_error(2 ** prec, 2 ** da, 2 ** db, 2 ** delta) for prec, da, db, delta in cotrans2_cases]
plt.bar([*map(str, cotrans2_cases)], errs)

# %%

prec = 2**-8
eps = 0.5 * prec
da = 2**-6
db = 2**-3
delta = 2**-3

xs = np.arange(-1, -db, prec)
exact = phi_sub(xs)
approx = cotrans3_rnd(prec, delta, da, db, xs)
err = max_err(exact, approx)
ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
err_bound = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
err, err_bound, err / err_bound * 100

# %%

def cotrans3_error(prec, da, db, delta):
    eps = 0.5 * prec
    xs = np.arange(-1, -db, prec)
    exact = phi_sub(xs)
    approx = cotrans3_rnd(prec, delta, da, db, xs)
    err = max_err(exact, approx)
    ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
    err_bound = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
    return err / err_bound * 100

cotrans3_cases = [
    # p, da, db, delta
    (-8, -6, -3, -3),
    (-8, -6, -3, -4),
    (-8, -5, -2, -3),
    (-8, -5, -2, -4),
    (-16, -12, -6, -4),
    (-16, -12, -6, -6),
    (-16, -10, -5, -4),
    (-16, -10, -5, -6),
    # (-18, -17, -15, -1)
]

errs = [cotrans3_error(2 ** prec, 2 ** da, 2 ** db, 2 ** delta) for prec, da, db, delta in cotrans3_cases]
plt.bar([*map(str, cotrans3_cases)], errs)

# %%

def cotrans_error(prec, da, db, delta):
    eps = 0.5 * prec
    xs2 = np.arange(-db, -da, prec)
    exact = phi_sub(xs2)
    approx = cotrans2_rnd(prec, delta, da, xs2)
    err2 = max_err(exact, approx)
    err_bound2 = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
    xs3 = np.arange(-1, -db, prec)
    exact = phi_sub(xs3)
    approx = cotrans3_rnd(prec, delta, da, db, xs3)
    err3 = max_err(exact, approx)
    ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
    err_bound3 = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_sub_rnd_err(prec, delta)
    return max(err2, err3) / max(err_bound2, err_bound3) * 100

cotrans_cases = [
    # p, da, db, delta
    (-8, -6, -3, -3),
    (-8, -6, -3, -4),
    (-8, -5, -2, -3),
    (-8, -5, -2, -4),
    (-16, -12, -6, -4),
    (-16, -12, -6, -6),
    (-16, -10, -5, -4),
    (-16, -10, -5, -6),
    # (-18, -17, -15, -1)
]

errs = [cotrans_error(2 ** prec, 2 ** da, 2 ** db, 2 ** delta) for prec, da, db, delta in cotrans_cases]
plt.bar([*map(str, cotrans_cases)], errs)
# %%
