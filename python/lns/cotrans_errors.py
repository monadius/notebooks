# %%

import numpy as np
from matplotlib import pyplot as plt
from definitions import *

# %%

def cotrans_error(prec: float, rounding_mode: RoundingMode, *, da: float, db: float, delta: float) -> tuple[float, float]:
    rnd = fix_rnd(prec, RoundingMode.NEAREST)
    nearest = rounding_mode == RoundingMode.NEAREST
    # Set the eps value
    if nearest:
        eps = 0.5 * prec
    else:
        eps = prec
    # Compute the fixed-point rounding error bound (for the approximation outside of (-1, 0])
    match rounding_mode:
        case RoundingMode.NEAREST | RoundingMode.FAITHFUL:
            taylor_rnd_bound = (2 + delta) * eps
        case RoundingMode.FLOOR | RoundingMode.CEIL:
            taylor_rnd_bound = (1 + delta) * eps
        case _: raise ValueError(f"Rounding mode {rounding_mode} not supported")
    taylor_err = taylor_sub_err_bound(delta) + taylor_rnd_bound
    # Make sure that theorem assumptions are satisfied
    assert 4 * eps <= da, f"da is too small: delta = {delta}, da = {da}, eps = {eps}"
    # This assertion is sufficient to ensure that phi_sub is always computed at x <= -1.
    # But it is not necessary. So it is commented out: there is an assertion in phi_sub which checks that x <= -1.
    # assert 8 * eps + 2 * taylor_err <= db, f"db is too small: delta = {delta}, da = {da}, db = {db}, eps = {eps}, taylor_err = {taylor_err}"
    # case 1 is skipped since the error bounds is eps which is smaller than case 2 or case 3 bounds
    # case 2
    xs2 = np.arange(-db, -da, prec)
    exact2 = phi_sub(xs2)
    approx2 = cotrans2_rnd(rnd, delta, da, xs2)
    err2 = max_err(exact2, approx2)
    err_bound2 = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_err
    # case 3
    xs3 = np.arange(-1, -db, prec)
    exact3 = phi_sub(xs3)
    approx3 = cotrans3_rnd(rnd, delta, da, db, xs3)
    err3 = max_err(exact3, approx3)
    ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_err
    err_bound3 = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_err
    return max(err2, err3), max(err_bound2, err_bound3)

