"""Exact physical support for the quintic fold and radius identities for the quartic example."""
import json
from fractions import Fraction
from math import prod
from pathlib import Path

from sympy import Poly, QQ, Rational, symbols


def quintic_at_fixed_scale():
    z, v, x, y, lam, mu = symbols("z v x y lambda mu")
    tau = Rational(1, 2**40)
    fold = z**4 * (1 + z)
    potential = fold**3 / 3 + tau**4 * fold**2 / 2
    scaled = z / tau
    primitive = -scaled + Rational(13, 18)*scaled**9 + Rational(19, 39)*scaled**13
    numerator = Poly(z**4 * (tau**4 + z**4) * primitive, z, domain=QQ)
    denominator = Poly(fold * (fold + tau**4), z, domain=QQ)
    h = (-numerator).rem(denominator)
    b = (numerator + h).exquo(denominator)
    assert (Poly(potential, z).degree(), denominator.degree(), numerator.degree(),
            h.degree(), b.degree()) == (15, 10, 21, 9, 11)
    local = Poly(z**4, z, domain=QQ)
    far = denominator.exquo(local)
    obstruction = Poly(fold + tau**4, z, domain=QQ)
    assert h.rem(local).is_zero
    assert far.gcd(obstruction) == obstruction
    assert not numerator.exquo(local).rem(obstruction).is_zero

    V = Poly((potential + lam*h.as_expr()).subs(z, v-1), v)
    F = Poly((fold + lam*b.as_expr()).subs(z, v-1), v)

    def coefficient_product(poly):
        # Lambda = 1 tests nonidentity of the coefficient polynomials only.
        coefficients = (poly.nth(i).subs(lam, 1) for i in range(1, poly.degree()+1))
        return prod(int(c.p) * pow(int(c.q), -1, 101) for c in coefficients) % 101

    assert pow(2**40, -1, 101) == 87
    assert (coefficient_product(V), coefficient_product(F)) == (71, 7)
    p = Poly(2*y*V.diff().as_expr().subs(v, y**2), x, y)
    q = Poly(-2*tau**12*x + mu*(F.as_expr().subs(v, y**2) - F.nth(0)), x, y)
    assert set(p.monoms()) == {(0, j) for j in range(1, 30, 2)}
    assert set(q.monoms()) == {(1, 0)} | {(0, j) for j in range(2, 23, 2)}
    assert p.coeff_monomial(y**29) == 10
    assert q.coeff_monomial(x) == -2*tau**12
    assert q.coeff_monomial(y**22) == 19*mu*lam/(39*tau**13)
    terms = len(p.monoms()) + len(q.monoms())
    degree = max(p.total_degree(), q.total_degree())
    assert (terms, degree, terms-1) == (27, 29, 26)

    wells = (terms-1)//2
    modes = terms-1-wells
    assert (wells*(modes-1), 2*max(wells, modes)-1) == (156, 25)
    print("tau = 2^-40. Degrees of A(J), D0, N0, H, B: 15, 10, 21, 9, 11.")
    print("Translated coefficient products at lambda = 1 modulo 101: 71, 7.")
    print("[y^29]P = 10; [x]Q = -2 tau^12; [y^22]Q = 19 mu lambda/(39 tau^13).")
    print("Exact support (15, 12), N = 27, D = 29, cap = 26.")
    print("Elementary field at N = 27: n = t = 13, lower bound 156, degree 25.")
    print("The residual factor J + tau^4 does not divide the prescribed numerator R.")


def published_radii():
    data = Path(__file__).resolve().parents[1] / "inputs" / "public.json"
    marks = json.loads(data.read_text())["bbjt"]
    # Bréhard, Mahboubi and Pous (2019), §6.2, Table 1, column r.
    radii = [Fraction(r) for r in ("0.5", "0.78", "0.88", "0.89", "0.895")]
    energies = [Fraction(h) for h in marks["small_h"]]
    assert [r*r for r in radii] == energies[:5]
    largest = max(energies + [Fraction(h) for h in marks["large_h"]])
    assert largest == Fraction(19, 20)**2
    assert Fraction(11, 10) - Fraction(19, 20) == Fraction(3, 20)
    print("Squares of the five published 2019 radii match the first five small-oval marks.")
    print("Largest transcribed energy = 361/400; disk bound y^2 >= 3/20.")


if __name__ == "__main__":
    quintic_at_fixed_scale()
    published_radii()
