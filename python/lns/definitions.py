import numpy as np
from typing import Callable, TypeAlias

Value: TypeAlias = np.ndarray
Rounding: TypeAlias = Callable[[Value], Value]

def max_err(exact: Value, approx: Value) -> float:
    return np.max(np.abs(exact - approx))

def fix_rnd(eps: float) -> Rounding: 
    return lambda xs: np.round(xs * (1 / eps)) * eps

# Φp and Φm and their derivatives

def phi_add(xs: Value) -> Value:
    return np.log2(1 + 2 ** xs)

def dphi_add(xs: Value) -> Value:
    return 2 ** xs / (2 ** xs + 1)

def phi_sub(xs: Value) -> Value:
    return np.log2(1 - 2 ** xs)

def dphi_sub(xs: Value) -> Value:
    return 2 ** xs / (2 ** xs - 1)

# First-order taylor approximations

def taylor_add(delta: float, xs: Value) -> Value:
    if np.any(xs > 0):
        raise ValueError('taylor_add: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_add(ns) - rs * dphi_add(ns)

# ΦTp_fix
def taylor_add_rnd(rnd: Rounding, delta: float, xs: Value) -> Value:
    if np.any(xs > 0):
        raise ValueError('taylor_add_rnd: xs > 0')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_add(ns)) - rnd(rs * rnd(dphi_add(ns)))

# Ep
def taylor_add_err(i: Value, r: Value) -> Value:
    return phi_add(i - r) - phi_add(i) + r * dphi_add(i)

# Ep_fix
def taylor_add_err_rnd(rnd: Rounding, i: Value, r: Value) -> Value:
    return phi_add(i - r) - rnd(phi_add(i)) + rnd(r * rnd(dphi_add(i)))

def taylor_add_err_bound(delta: float) -> float:
    return taylor_add_err(0, delta)

def taylor_add_rnd_err_bound(prec: float, delta: float) -> float:
    eps = 0.5 * prec
    return taylor_add_err_bound(delta) + (2 + delta) * eps

def taylor_sub(delta: float, xs: Value) -> Value:
    if np.any(xs > -1):
        raise ValueError('taylor_sub: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return phi_sub(ns) - rs * dphi_sub(ns)

# ΦTm_fix
def taylor_sub_rnd(rnd: Rounding, delta: float, xs: Value) -> Value:
    if np.any(xs > -1):
        raise ValueError('taylor_sub_rnd: xs > -1')
    ns = np.ceil(xs / delta) * delta
    rs = ns - xs
    return rnd(phi_sub(ns)) - rnd(rs * rnd(dphi_sub(ns)))

# Em
def taylor_sub_err(i: Value, r: Value) -> Value:
    return -phi_sub(i - r) + phi_sub(i) - r * dphi_sub(i)

# Em_fix
def taylor_sub_err_rnd(rnd: Rounding, i: Value, r: Value) -> Value:
    return phi_sub(i - r) - rnd(phi_sub(i)) + rnd(r * rnd(dphi_sub(i)))

def taylor_sub_err_bound(delta: float) -> float:
    return taylor_sub_err(-1, delta)

def taylor_sub_rnd_err_bound(prec: float, delta: float) -> float:
    eps = 0.5 * prec
    return taylor_sub_err_bound(delta) + (2 + delta) * eps

# Co-transformations

def ind(delta: float, xs: Value) -> Value:
    return (np.ceil(xs / delta) - 1) * delta

def rem(delta: float, xs: Value) -> Value:
    return ind(delta, xs) - xs

def kval(delta: float, xs: Value) -> Value:
    return xs - phi_sub(ind(delta, xs)) + phi_sub(rem(delta, xs))

def k_rnd(rnd: Rounding, delta: float, xs: Value) -> Value:
    return xs - rnd(phi_sub(ind(delta, xs))) + rnd(phi_sub(rem(delta, xs)))

def cotrans2(delta: float, da: float, xs: Value) -> Value:
    return phi_sub(ind(da, xs)) + taylor_sub(delta, kval(da, xs))

def cotrans2_rnd(rnd: Rounding, delta: float, da: float, xs: Value) -> Value:
    return rnd(phi_sub(ind(da, xs))) + taylor_sub_rnd(rnd, delta, k_rnd(rnd, da, xs))

def cotrans3(delta: float, da: float, db: float, xs: Value) -> Value:
    return phi_sub(ind(db, xs)) + taylor_sub(delta, kval(db, xs))

def cotrans3_rnd(rnd: Rounding, delta: float, da: float, db: float, xs: Value) -> Value:
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


