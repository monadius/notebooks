# %%
from sympy import *

x, i, j, r, Δ, ΔP, ε, t = symbols('x i j r Δ ΔP ε t')

# %%
Φp = Lambda(x, log(1 + 2 ** x, 2))
Φm = Lambda(x, log(1 - 2 ** x, 2))
# %%

a, b = symbols('a b')

Φm(a + b)
# %%

f = Φm(b) + Φp(b + Φm(a) - Φm(b))
# %%
simplify(f - Φm(a + b))
# %%
