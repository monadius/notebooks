import numpy as np

def max_err(exact, approx):
    return np.max(np.abs(exact - approx))

def fix_rnd(eps): 
    return lambda xs: np.round(xs * (1 / eps)) * eps

# Φp and Φm and their derivatives

def phi_add(xs):
    return np.log2(1 + 2 ** xs)

def dphi_add(xs):
    return 2 ** xs / (2 ** xs + 1)

def phi_sub(xs):
    return np.log2(1 - 2 ** xs)

def dphi_sub(xs):
    return 2 ** xs / (2 ** xs - 1)

# First-order taylor approximations

def taylor_add(delta, xs):
    if np.any(xs > 0):
        raise ValueError('taylor_add: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_add(ns) - rs * dphi_add(ns)

# ΦTp_fix
def taylor_add_rnd(rnd, delta, xs):
    if np.any(xs > 0):
        raise ValueError('taylor_add_rnd: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))

# Ep
def taylor_add_err(i, r):
    return phi_add(i - r) - phi_add(i) + r * dphi_add(i)

# Ep_fix
def taylor_add_err_rnd(rnd, i, r):
    return phi_add(i - r) - rnd(phi_add(i)) + rnd(r * rnd(dphi_add(i)))

def taylor_add_err_bound(delta):
    return taylor_add_err(0, delta)

def taylor_add_rnd_err(prec, delta):
    eps = 0.5 * prec
    return taylor_add_err_bound(delta) + (2 + delta) * eps

def taylor_sub(delta, xs):
    if np.any(xs > -1):
        raise ValueError('taylor_sub: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_sub(ns) - rs * dphi_sub(ns)

# ΦTm_fix
def taylor_sub_rnd(rnd, delta, xs):
    if np.any(xs > -1):
        raise ValueError('taylor_sub_rnd: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_sub(ns)) - rnd(rs * rnd(dphi_sub(ns)))

# Em
def taylor_sub_err(i, r):
    return -phi_sub(i - r) + phi_sub(i) - r * dphi_sub(i)

# Em_fix
def taylor_sub_err_rnd(rnd, i, r):
    return phi_sub(i - r) - rnd(phi_sub(i)) + rnd(r * rnd(dphi_sub(i)))

def taylor_sub_err_bound(delta):
    return taylor_sub_err(-1, delta)

def taylor_sub_rnd_err(prec, delta):
    eps = 0.5 * prec
    return taylor_sub_err_bound(delta) + (2 + delta) * eps

# Co-transformations

def ind(delta, xs):
    return (np.ceil(xs / delta) - 1) * delta

def rem(delta, xs):
    return ind(delta, xs) - xs

def kval(delta, xs):
    return xs - phi_sub(ind(delta, xs)) + phi_sub(rem(delta, xs))

def k_rnd(rnd, delta, xs):
    return xs - rnd(phi_sub(ind(delta, xs))) + rnd(phi_sub(rem(delta, xs)))

def cotrans2(delta, da, xs):
    return phi_sub(ind(da, xs)) + taylor_sub(delta, kval(da, xs))

def cotrans2_rnd(rnd, delta, da, xs):
    return rnd(phi_sub(ind(da, xs))) + taylor_sub_rnd(rnd, delta, k_rnd(rnd, da, xs))

def cotrans3(delta, da, db, xs):
    return phi_sub(ind(db, xs)) + taylor_sub(delta, kval(db, xs))

def cotrans3_rnd(rnd, delta, da, db, xs):
    rab = rem(db, xs)
    res = np.zeros(len(xs))
    special = rab >= -da
    incl = rab < -da
    rab, xs, ys = rab[incl], xs[incl], xs[special]
    rb = ind(da, rab)
    k1 = k_rnd(rnd, da, rab)
    k2 = xs + rnd(phi_sub(rb)) + taylor_sub_rnd(rnd, delta, k1) - rnd(phi_sub(ind(db, xs)))
    res[incl] = rnd(phi_sub(ind(db, xs))) + taylor_sub_rnd(rnd, delta, k2)
    res[special] = cotrans2_rnd(rnd, delta, db, ys)
    return res

