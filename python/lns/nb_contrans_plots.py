# %%

import numpy as np
from matplotlib import pyplot as plt
from definitions import *
import taylor_errors as te
import cotrans_errors as ce


# Contrasformation errors only
# %%

# rounding_mode = RoundingMode.FAITHFUL
rounding_mode = RoundingMode.FLOOR
prec = -18
da = 2 ** -15
db = 2 ** -8

# prec = -20
# da = 2 ** -18
# db = 2 ** -9

deltas = [*range(-12, -3)]
xy = []
xy.append((deltas, r'Cotransformation', [ce.cotrans_error(2 ** prec, rounding_mode, da=da, db=db, delta=2 ** d) for d in deltas]))

# %%
fig = plt.figure(figsize = (14, 10))
linewidth = 3
fontsize = 16
for (ds, label, errs), color in zip(xy, ('green', 'red', 'blue')):
    plt.plot(ds, np.log2([err[0] for err in errs]), color=color, linewidth=linewidth, label='Actual error bound')
    plt.plot(ds, np.log2([err[1] for err in errs]), color=color, linewidth=linewidth, linestyle='--', label='Theoretical error bound')
plt.xlabel(r'$\log_2(\Delta)$', fontsize=fontsize + 2)
plt.ylabel(r'$\log_2(\rm{error})$', fontsize=fontsize + 2)
plt.xticks(deltas, fontsize=fontsize)
plt.yticks(range(-18, -3), fontsize=fontsize)
plt.legend(loc='upper left', fontsize=fontsize + 5)
plt.title(fr'Error bounds for cotransformation $\epsilon = 2^{{{prec}}}, \Delta_a = 2^{{{np.log2(da)}}}, \Delta_b = 2^{{{np.log2(db)}}}$', fontsize=fontsize + 5)
# plt.savefig('images/taylor_add_sub_cotrans_floor.png', bbox_inches='tight')
# plt.savefig('images/taylor_add_sub_cotrans.pdf', bbox_inches='tight', format='pdf')



# Contrasformation 2 errors only
# %%


# rounding_mode = RoundingMode.FAITHFUL
rounding_mode = RoundingMode.FLOOR
prec = -18
da = 2 ** -9

# prec = -20
# da = 2 ** -18
# db = 2 ** -9

deltas = [*range(-12, -3)]
xy = []
xy.append((deltas, r'Cotransformation', [ce.cotrans_error2(2 ** prec, rounding_mode, da=da, delta=2 ** d) for d in deltas]))

# %%
fig = plt.figure(figsize = (14, 10))
linewidth = 3
fontsize = 16
for (ds, label, errs), color in zip(xy, ('green', 'red', 'blue')):
    plt.plot(ds, np.log2([err[0] for err in errs]), color=color, linewidth=linewidth, label='Actual error bound')
    plt.plot(ds, np.log2([err[1] for err in errs]), color=color, linewidth=linewidth, linestyle='--', label='Theoretical error bound')
plt.xlabel(r'$\log_2(\Delta)$', fontsize=fontsize + 2)
plt.ylabel(r'$\log_2(\rm{error})$', fontsize=fontsize + 2)
plt.xticks(deltas, fontsize=fontsize)
plt.yticks(range(-18, -3), fontsize=fontsize)
plt.legend(loc='upper left', fontsize=fontsize + 5)
plt.title(fr'Error bounds for 2-table cotransformation $\epsilon = 2^{{{prec}}}, \Delta_a = 2^{{{np.log2(da)}}}$', fontsize=fontsize + 5)
# plt.savefig('images/taylor_add_sub_cotrans_floor.png', bbox_inches='tight')
# plt.savefig('images/taylor_add_sub_cotrans.pdf', bbox_inches='tight', format='pdf')


# Contrasformation 2 errors only for fixed delta and varying da
# %%


# rounding_mode = RoundingMode.FAITHFUL
rounding_mode = RoundingMode.FLOOR
prec = -18
delta = 2 ** -5

# prec = -20
# da = 2 ** -18
# db = 2 ** -9

das = [*range(-16, -4)]
xy = []
xy.append((das, r'Cotransformation', [ce.cotrans_error2(2 ** prec, rounding_mode, da=2 ** da, delta=delta) for da in das]))

# %%
fig = plt.figure(figsize = (14, 10))
linewidth = 3
fontsize = 16
for (ds, label, errs), color in zip(xy, ('green', 'red', 'blue')):
    plt.plot(ds, np.log2([err[0] for err in errs]), color=color, linewidth=linewidth, label='Actual error bound')
    plt.plot(ds, np.log2([err[1] for err in errs]), color=color, linewidth=linewidth, linestyle='--', label='Theoretical error bound')
plt.xlabel(r'$\log_2(\Delta_a)$', fontsize=fontsize + 2)
plt.ylabel(r'$\log_2(\rm{error})$', fontsize=fontsize + 2)
plt.xticks(das, fontsize=fontsize)
plt.yticks(range(-18, -3), fontsize=fontsize)
plt.legend(loc='upper left', fontsize=fontsize + 5)
plt.title(fr'Error bounds for 2-table cotransformation for varying $\Delta_a$ with $\epsilon = 2^{{{prec}}}, \Delta = 2^{{{np.log2(delta)}}}$', fontsize=fontsize + 5)
# plt.savefig('images/taylor_add_sub_cotrans_floor.png', bbox_inches='tight')
# plt.savefig('images/taylor_add_sub_cotrans.pdf', bbox_inches='tight', format='pdf')


# %%
