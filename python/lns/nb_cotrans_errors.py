# %%

import numpy as np
from matplotlib import pyplot as plt
from definitions import *

# %%

prec = 2 ** -24
eps = 0.5 * prec

da = 2 ** -20
db = 2 ** -10
delta = 2 ** -3

print(da >= 4 * eps, db >= 8 * eps + delta ** 2 * np.log(2))
xs = np.arange(-1, -db, prec)
ks = kval(db, xs)
ks[ks >= -1]


# %%

prec = 2**-8
da = 2**-6
db = 2**-3
xs = np.arange(-1, -db, prec)
ys = xs[rem(db, xs) >= -da]
kval(da, rem(db, xs)) >= -1

# %%

prec = 2**-8
eps = 0.5 * prec
da = 2**-6
db = 2**-3
delta = 2**-3
rnd = fix_rnd(prec, RoundingMode.NEAREST)

xs = np.arange(-db, -da, prec)
exact = phi_sub(xs)
approx = cotrans2_rnd(rnd, delta, da, xs)
err = max_err(exact, approx)
err_bound = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
err, err_bound, err / err_bound * 100

# %%

def cotrans2_error(prec: float, da: float, db: float, delta: float) -> float:
    eps = 0.5 * prec
    rnd = fix_rnd(prec, RoundingMode.NEAREST)
    xs = np.arange(-db, -da, prec)
    exact = phi_sub(xs)
    approx = cotrans2_rnd(rnd, delta, da, xs)
    err = max_err(exact, approx)
    err_bound = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
    return err / err_bound * 100

cotrans2_cases: list[tuple[float, float, float, float]] = [
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
plt.xticks(rotation=20, fontsize=8)
plt.xlabel("prec, da, db, delta")
plt.ylabel("error (%)")
plt.figtext(0.5, -0.1, "cotrans2 errors", ha="center", fontsize=14)
plt.tight_layout()
plt.show()
# %%

prec = 2**-8
eps = 0.5 * prec
da = 2**-6
db = 2**-3
delta = 2**-3
rnd = fix_rnd(prec, RoundingMode.NEAREST)

xs = np.arange(-1, -db, prec)
exact = phi_sub(xs)
approx = cotrans3_rnd(rnd, delta, da, db, xs)
err = max_err(exact, approx)
ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
err_bound = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
err, err_bound, err / err_bound * 100

# %%

def cotrans3_error(prec: float, da: float, db: float, delta: float) -> float:
    eps = 0.5 * prec
    rnd = fix_rnd(prec, RoundingMode.NEAREST)
    xs = np.arange(-1, -db, prec)
    exact = phi_sub(xs)
    approx = cotrans3_rnd(rnd, delta, da, db, xs)
    err = max_err(exact, approx)
    ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
    err_bound = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
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
plt.show()
# %%

def cotrans_error(prec: float, da: float, db: float, delta: float) -> float:
    eps = 0.5 * prec
    rnd = fix_rnd(prec, RoundingMode.NEAREST)
    xs2 = np.arange(-db, -da, prec)
    exact = phi_sub(xs2)
    approx = cotrans2_rnd(rnd, delta, da, xs2)
    err2 = max_err(exact, approx)
    err_bound2 = eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
    xs3 = np.arange(-1, -db, prec)
    exact = phi_sub(xs3)
    approx = cotrans3_rnd(rnd, delta, da, db, xs3)
    err3 = max_err(exact, approx)
    ek2 = 2 * eps + (phi_sub(-1 - 2 * eps) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
    err_bound3 = eps + (phi_sub(-1 - ek2) - phi_sub(-1)) + taylor_sub_rnd_err_bound(prec, delta)
    return max(err2, err3) / max(err_bound2, err_bound3) * 100

cotrans_cases: list[tuple[float, float, float, float]] = [
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
plt.show()
# %%

import numpy as np
from matplotlib import pyplot as plt
from definitions import *
import cotrans_errors as ce

ce.cotrans_error(2**-15, RoundingMode.NEAREST, da=2**-6, db=2**-5, delta=2**-3)
# %%

rounding_mode = RoundingMode.FAITHFUL
prec = -23
da = 2 ** -20
db = 2 ** -10
deltas = [*range(-13, 0)]
xy = []
xy.append((deltas, r'Co-transformation', [ce.cotrans_error(2 ** prec, rounding_mode, da=da, db=db, delta=2 ** d) for d in deltas]))

# %%
fig = plt.figure(figsize = (14, 10))
linewidth = 3
fontsize = 16
for (ds, label, errs), color in zip(xy, ('red', 'green', 'blue')):
    plt.plot(ds, np.log2([err[0] for err in errs]), color=color, linewidth=linewidth, label=label)
    plt.plot(ds, np.log2([err[1] for err in errs]), color=color, linewidth=linewidth, linestyle='--')
plt.xlabel(r'$\log_2(\Delta)$', fontsize=fontsize + 2)
plt.ylabel(r'$\log_2(\rm{error})$', fontsize=fontsize + 2)
plt.xticks(range(-13, 0), fontsize=fontsize)
plt.yticks(range(-23, -4, 2), fontsize=fontsize)
plt.legend(loc='lower right', fontsize=fontsize + 5)

# %%
