"""Algebra for Hermite interpolation and replication of a given finite seed."""
from itertools import product

import sympy as sp


def hermite_polynomial(v, nodes):
    """The value basis vanishes and every prescribed first derivative is one."""
    terms = []
    for node in nodes:
        basis = sp.prod((v - other) / (node - other) for other in nodes if other != node)
        terms.append((v - node) * basis**2)
    return sp.Poly(sum(terms), v, domain=sp.QQ)


def check_charts():
    u, v, x, a = sp.symbols("u v x a")
    f = 3 - 2*x + 3*x**2
    g = 2 + x - x**2
    seed = sp.Matrix([v - f - a*g, g])
    assert sp.expand(seed.row_join(seed.diff(a)).det() - g**2) == 0
    for k in range(1, 6):
        nodes = [sp.Rational(2*j - k + 1, 2) for j in range(k)]
        h = hermite_polynomial(v, nodes)
        assert h.degree() <= 2*k - 1 and all(i % 2 for i, in h.monoms())
        assert all(h.eval(node) == 0 and h.diff().eval(node) == 1 for node in nodes)
        for t, n in product((2, 4, 6), (1, 2, 3)):
            phi = 3 * sp.chebyshevt(t, u)
            test_f = sum((j+1)*x**j for j in range(n+1))
            test_g = sum((3*j-2)*x**j for j in range(n+1))
            p = sp.Poly(h.as_expr() - test_f.subs(x, phi), u, v)
            q = sp.Poly(sp.diff(phi, u) * test_g.subs(x, phi), u, v)
            chart = sp.Matrix([phi, v])
            image = chart.jacobian([u, v]) * sp.Matrix([p.as_expr(), q.as_expr()])
            strip = sp.Matrix([h.as_expr()-test_f, test_g]).subs(x, phi)
            assert (image-phi.diff(u)*strip).applyfunc(sp.expand) == sp.zeros(2, 1)
            assert all((i == 0 and j % 2 == 1) or (j == 0 and i % 2 == 0)
                       for i, j in p.monoms())
            assert all(i % 2 == 1 and j == 0 for i, j in q.monoms())
            slots = len(p.terms()) + len(q.terms())
            assert slots <= k + t*n + t//2 + 1 <= k + t*(n+1)
            assert max(p.total_degree(), q.total_degree()) <= max(2*k-1, t*(n+1)-1)
            assert p.gcd(q).total_degree() == 0
    for order, direction in product(range(1, 7), (-1, 1)):
        for level in (-1, 1):
            equation = sp.Poly(direction*x**order - sp.Rational(level, 2), x)
            roots = equation.count_roots(-1, 1)
            assert roots == (1 if order % 2 else 2*int(level == direction))
            assert equation.gcd(equation.diff()).degree() == 0
    for odd, plus, minus in product(range(8), repeat=3):
        larger = max(odd+2*plus, odd+2*minus)
        assert larger == odd+plus+minus+abs(plus-minus)
    print("Odd Hermite jets at k=1..5, parity, degree, gcd and term ceilings for 45 polynomial pullbacks.")
    print("Rotated-family determinant G^2 and both sign counts checked; hyperbolicity follows from the return-map proof.")


if __name__ == "__main__":
    check_charts()
