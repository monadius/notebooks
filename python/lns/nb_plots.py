# %%

import numpy as np
from matplotlib import pyplot as plt
from definitions import *
import taylor_errors as te
import cotrans_errors as ce

# %%

delta = 2 ** -0
prec = 2 ** -10
xs = np.arange(-6 * delta, 0, prec)
exact = phi_add(xs)
approx = taylor_add(delta, xs)
error = exact - approx

# %%

linewidth = 2.5
fontsize = 16

fig, ax = plt.subplots(figsize=(14, 7))

# The main plot
ax.plot(xs, error, color='blue', linewidth=linewidth, label=r'$E_T(x) = \Phi^+(x) - \hat{\Phi}^+_T(x)$')

# Red dots
x_points = [-delta + 1e-10, -2 * delta + 1e-10, -3 * delta + 1e-10]
y_points = [phi_add(x) - taylor_add(delta, x) for x in x_points]
labels = [r'$E_\Delta(0)$', r'$E_{\Delta}(-\Delta)$', r'$E_{\Delta}(-2\Delta)$']
ax.scatter(x_points, y_points, color='red', s=100, linewidth=2)  # s is the size of the dots
for (x, y, label) in zip(x_points, y_points, labels):
    plt.text(x, y + 0.004, label, fontsize=fontsize, ha='center')  # ha controls horizontal alignment

# Spines and ticks
ax.spines[["right", "bottom"]].set_position(("data", 0))
ax.spines[["right", "bottom"]].set_linewidth(2)
ax.spines[["top", "left"]].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

ax.yaxis.tick_right()
ax.tick_params(axis='both', width=2)  # Set tick width for both axes
plt.xticks(
    ticks=[-d * delta for d in range(1, 7)], 
    labels=[fr'${-d if d != 1 else "-"}\Delta$' for d in range(1, 7)],
    fontsize=fontsize
)
plt.yticks([n * 0.01 for n in range(2, 9, 2)], fontsize=fontsize)

ax.legend(loc='upper left', fontsize=fontsize + 5, frameon=False)
plt.savefig('images/taylor_add_error.pdf', bbox_inches='tight', format='pdf')



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

te.plot_taylor_error_bar(
    'add',
    test_cases, 
    RoundingMode.NEAREST, 
    plot_improved=True,
    title='Taylor Addition (rounding to nearest)', 
    filename='images/taylor_add_err_nearest.png'
)

te.plot_taylor_error_bar(
    'add',
    test_cases, 
    RoundingMode.FLOOR, 
    title='Taylor Addition (directed rounding)', 
    filename='images/taylor_add_err_directed.png'
)

# %%

te.plot_taylor_error_bar(
    'sub',
    test_cases, 
    RoundingMode.NEAREST, 
    title='Taylor Subtraction (rounding to nearest)', 
    filename='images/taylor_sub_err_nearest.png'
)

te.plot_taylor_error_bar(
    'sub',
    test_cases, 
    RoundingMode.FLOOR, 
    title='Taylor Subtraction (directed rounding)', 
    filename='images/taylor_sub_err_directed.png'
)

# te.plot_taylor_error_bar(
#     'add',
#     test_cases, 
#     RoundingMode.FAITHFUL, 
#     'Taylor Addition (faithful rounding)', 
#     'images/taylor_add_err_directed.png'
# )

# %%
te.plot_error(-10, 'add', RoundingMode.FLOOR)
te.plot_error(-15, 'add', RoundingMode.FLOOR)
# te.plot_error(-23, 'add', RoundingMode.FLOOR)
# te.plot_error(-20, 'add', RoundingMode.FLOOR)
# te.plot_error(-24, 'add', RoundingMode.FLOOR)

# %%

te.plot_error(-10, 'add', RoundingMode.FAITHFUL)
te.plot_error(-15, 'add', RoundingMode.FAITHFUL)


# %%
te.plot_error(-10, 'add', RoundingMode.NEAREST)
te.plot_error(-15, 'add', RoundingMode.NEAREST)
# te.plot_error(-23, 'add', RoundingMode.NEAREST)


# %%
te.plot_error(-10, 'sub', RoundingMode.FLOOR)
te.plot_error(-15, 'sub', RoundingMode.FLOOR)
# te.plot_error(-23, 'add', RoundingMode.FLOOR)
# te.plot_error(-20, 'add', RoundingMode.FLOOR)
# te.plot_error(-24, 'add', RoundingMode.FLOOR)

# %%
te.plot_error(-10, 'sub', RoundingMode.NEAREST)
te.plot_error(-15, 'sub', RoundingMode.NEAREST)
# plot_error(-23, 'add', RoundingMode.NEAREST)

# %%
te.plot_error(-10, 'add average', RoundingMode.NEAREST)
te.plot_error(-15, 'add average', RoundingMode.NEAREST)

# %%
te.plot_error(-10, 'add average', RoundingMode.FLOOR)
te.plot_error(-15, 'add average', RoundingMode.FLOOR)

# %%
rounding_mode = RoundingMode.FAITHFUL
precs = [-10, -15, -23]
deltas = [*range(-13, 0)]
xy = []
for prec in precs:
    # ds = [d for d in deltas if d >= prec + 1]
    ds = deltas
    xy.append((ds, [te.get_add_error(2 ** prec, 2 ** d, rounding_mode) for d in ds]))

# %%
fig = plt.figure(figsize = (14, 10))
linewidth = 3
fontsize = 16
for prec, (ds, errs), color in zip(precs, xy, ('blue', 'green', 'red')):
    plt.plot(ds, np.log2([err[0] for err in errs]), color=color, linewidth=linewidth, label=fr'$\epsilon = 2^{{{prec}}}$')
    plt.plot(ds, np.log2([err[1] for err in errs]), color=color, linewidth=linewidth, linestyle='--')
plt.xlabel(r'$\log_2(\Delta)$', fontsize=fontsize + 2)
plt.ylabel(r'$\log_2(\rm{error})$', fontsize=fontsize + 2)
plt.xticks(range(-13, 0), fontsize=fontsize)
plt.yticks(range(-23, -4, 2), fontsize=fontsize)
plt.legend(loc='lower right', fontsize=fontsize + 5)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.savefig('images/taylor_error1.png', bbox_inches='tight')


# %%

rounding_mode = RoundingMode.FAITHFUL
prec = -23
da = 2 ** -20
db = 2 ** -10
deltas = [*range(-15, -5)]
xy = []
xy.append((deltas, r'Taylor $\Phi^+$', [te.get_add_error(2 ** prec, 2 ** d, rounding_mode) for d in deltas]))
xy.append((deltas, r'Taylor $\Phi^-$', [te.get_sub_error(2 ** prec, 2 ** d, rounding_mode) for d in deltas]))
xy.append((deltas, r'Co-transformation', [ce.cotrans_error(2 ** prec, rounding_mode, da=da, db=db, delta=2 ** d) for d in deltas]))

# %%
fig = plt.figure(figsize = (14, 10))
linewidth = 3
fontsize = 16
for (ds, label, errs), color in zip(xy, ('green', 'red', 'blue')):
    plt.plot(ds, np.log2([err[0] for err in errs]), color=color, linewidth=linewidth, label=label)
    plt.plot(ds, np.log2([err[1] for err in errs]), color=color, linewidth=linewidth, linestyle='--')
plt.xlabel(r'$\log_2(\Delta)$', fontsize=fontsize + 2)
plt.ylabel(r'$\log_2(\rm{error})$', fontsize=fontsize + 2)
plt.xticks(deltas, fontsize=fontsize)
plt.yticks(range(-23, -11), fontsize=fontsize)
plt.legend(loc='upper left', fontsize=fontsize + 5)
# plt.savefig('images/taylor_add_sub_cotrans2.png', bbox_inches='tight')
plt.savefig('images/taylor_add_sub_cotrans2.pdf', bbox_inches='tight', format='pdf')


# %%
