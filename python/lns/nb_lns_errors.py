# %%
from sympy import *

x, i, r, Δ, ΔP, ε = symbols('x i r Δ ΔP ε')

# %%
Φp = Lambda(x, log(1 + 2 ** x, 2))
Φm = Lambda(x, log(1 - 2 ** x, 2))

# %%
Ep = Lambda((i, r), Φp(i - r) - Φp(i) + r * diff(Φp(i), i))
Em = Lambda((i, r), -Φm(i - r) + Φm(i) - r * diff(Φm(i), i))

# %%
err_p = Ep(0, Δ).series(Δ)
err_m = Em(-1, Δ).series(Δ)
print(err_p.evalf())
err_p

# %%
print(err_m.evalf())
err_m

# %%
Qp = Lambda((i, r), Ep(i, r) / Ep(i, Δ))
Qp_lo = Qp(0, r)
Qp_hi = (2 ** -r + r * log(2) - 1) / (2 ** -Δ + Δ * log(2) - 1)
A = Lambda(x, -2 * x * (log(x + 1) - log(x) - log(2)) - x + 1)
B = Lambda(x, x * (2 * log(x + 1) - log(x) - 2 * log(2)))
Rp_opt = log(B(2 ** Δ) / A(2 ** Δ), 2)

QRp = Qp_hi.subs(r, Rp_opt) - Qp_lo.subs(r, Rp_opt)
QIp = 1 - Qp_lo.subs(r, Δ - ΔP)

# %%
print(QRp.subs(Δ, 0.001).evalf())
print(QIp.subs(Δ, 0.00001).subs(ΔP, 0.00000005).evalf())

# %%
QIp.subs(ΔP, Δ / 256).series().evalf()

# %%
QIp.subs(ΔP, Δ * r).series(Δ, 0, 2).evalf()

# %%
A(x).series(x, 1)

# %%
B(x).series(x, 1)

# %%
Rp_opt.series(Δ)
# %%
expr = Qp_hi.subs(r, 2 * Δ / 3 - Δ**2 * log(2) / 36) - Qp_lo.subs(r, 2 * Δ / 3 - Δ**2 * log(2) / 36)
print(expr.series(Δ).evalf())
expr.series(Δ)
# %%
d1 = diff(Δ ** 2 * log(2) / 8 - Ep(0, Δ), Δ)
d1
# %%
plot(d1, (Δ, 0, 1))
# %%
d2 = diff(d1, Δ)
d2.simplify()
# %%
Em(-1 + r, Δ).series(Δ, 0, 3)

# %%
t = symbols('t')
expr = 8 * log(2) * t ** 2 / (16 * t ** 2 - 64 * t + 64) - 2 * log(2) * t / (4 * t - 8)
expr.simplify()
# %%
(2 ** t).series(t)
# %%
plot(2 ** r * log(2) / (2 ** r - 2)**2, (r, 0, 0.5))
# %%
