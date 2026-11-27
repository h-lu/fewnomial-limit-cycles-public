"""Exact checks for the quintic transfer, the loss audit and the literature slot count."""
from fractions import Fraction
from math import prod

from flint import fmpq as Q, fmpq_poly as Poly
from sympy import Poly as SymPoly
from sympy import QQ, chebyshevt, diff, expand, integrate, symbols

from finite import endpoint_count


def quintic_transfer():
    z = Poly([0, 1])
    one = Poly([1])
    fold = z**4 * (1 + z)
    psi = z**8 / 2 + z**12 / 3
    primitive = (psi + 3 * z * psi.derivative() / 2 - 1).integral()
    assert primitive == -z + Q(13, 18) * z**9 + Q(19, 39) * z**13
    # At E in [3/4, 1], A'/A <= 5/(8E); above 1 the slope is at least A.
    left, right = Q(3, 4), Q(5, 4)
    assert left - 1 == -Q(1, 4) and right - 1 == Q(1, 4)
    assert 1 - (1 - left) * Q(5, 8) / left == Q(19, 24)
    assert psi(Q(2)) > Q(3, 2)
    assert fold.degree() == 5 and fold.derivative() == z**3 * (4 + 5*z)

    fixed_scale = Q(1, 2**40)
    for tau in (Q(1, 8), Q(1, 16), Q(1, 32), fixed_scale):
        potential = fold**3 / 3 + tau**4 * fold**2 / 2
        denominator = fold * (fold + tau**4)
        scaled = z / tau
        numerator = z**4 * tau**4 * (1 + scaled**4) * primitive(scaled)
        h = (-numerator) % denominator
        b, rem = divmod(numerator + h, denominator)
        assert not rem and denominator * b - h == numerator and not (h % z**4)
        assert (potential.degree(), denominator.degree(), numerator.degree(), b.degree()) == (15, 10, 21, 11)
        assert h.degree() < 10 and denominator.gcd(one) == one
        far = denominator // z**4
        target = numerator // z**4
        assert far(tau*z) / tau**4 == (1 + tau*z) * (1 + z**4 * (1 + tau*z))
        assert target(tau*z) / tau**4 == (1 + z**4) * primitive
        assert potential(tau*z) / tau**12 == z**12*(1 + tau*z)**3/3 + z**8*(1 + tau*z)**2/2
        assert 1 - 2*tau >= Q(1, 2)
        assert not (target % z) and target % z**2  # missing C leaves order -3
        # A drive-only numerator is D_0 B. This target is not divisible by D_0.
        assert h and numerator % denominator
        # Change the inherited derivative to J+tau^4. Local disks and D0 stay fixed.
        shared_far = fold + tau**4
        assert far.gcd(shared_far) == shared_far and target % shared_far
        assert shared_far(-2) < 0 < shared_far(-1)
        # A shared *cancelled* factor can be allowed: C=J, G=J+tau^4, E0=J.
        # Test a basis of all admissible targets R of degree <= deg G + 11.
        unit, _, inverse = shared_far.xgcd(fold)
        assert unit == one
        for power in range(shared_far.degree() + 12):
            rhs = z**power
            reduced_h = (-rhs * inverse) % shared_far
            drive, remainder = divmod(rhs + fold*reduced_h, shared_far)
            assert not remainder and drive.degree() <= 11
            assert denominator*drive - fold*(fold*reduced_h) == fold*rhs

    # At the last scale, no generic padding is needed. A single value mod 101
    # detects each affine coefficient in lambda as a nonzero polynomial.
    def product_mod_101(poly):
        translated = poly(z - 1)
        residues = [int(c.p) * pow(int(c.q), -1, 101) for c in list(translated)[1:]]
        return prod(residues) % 101

    assert tau == fixed_scale and pow(2**40, -1, 101) == 87
    assert (product_mod_101(potential + h), product_mod_101(fold + b)) == (71, 7)

    # Exact upper bounds in the inverse chart w=Z(1+tau Z)^(1/4) from the quintic-fold proposition.
    tau_max = Q(1, 8)
    assert 2**4 * (1 - 2*tau_max) > Q(3, 2)**4 and psi(Q(3, 2)) > Q(3, 2)
    assert (Q(3, 4)**3) * 2**4 > 1  # u^(-3/4)<2
    assert Q(5, 4) * (1 + 2*32 + 1) <= 83
    assert Q(9, 2) / Q(9, 16) <= 9 and Q(83) / Q(9, 16)**2 <= 512
    for derivative in (primitive, primitive.derivative(), primitive.derivative().derivative()):
        assert sum(abs(c) * 2**i for i, c in enumerate(derivative)) <= 2**19
    assert Q(2**19 * (512+9+2+2), 1) / Q(3, 4) < 2**30
    error = fixed_scale * 2**30
    assert error == Q(1, 1024) and error < Q(1, 4)
    slope_margin = Q(19, 24) - error * Q(5, 8) / left
    assert slope_margin == Q(4859, 6144) and slope_margin > 0
    p_support = {(0, 2*i-1) for i in range(1, 16)}
    q_support = {(1, 0)} | {(0, 2*j) for j in range(1, 12)}
    assert (len(p_support), len(q_support), max(i+j for i,j in p_support | q_support)) == (15, 12, 29)
    print("Quintic J=z^4(1+z), D0=J(J+tau^4), C=z^4, E0=1.")
    print("Exact divisions and rescaled identities at tau=1/8, 1/16, 1/32, 1/2^40.")
    print("deg N0=21, deg D0=10, deg H<=9, deg B=11, deg S=13. Support (15,12), D=29 at tau=1/2^40 outside finitely many lambda.")
    print("Model normalized endpoint values -1/4, 1/4. Model slope/area >= 19/24.")
    print("tau=1/2^40: interval endpoint margins/area >= 255/1024; slope/area >=", slope_margin)
    print("Coefficient products at lambda=1 modulo 101: 71 and 7. One common small lambda, then mu, by the quintic-fold proposition.")
    print("Shared far factor J+tau^4 obstructs the target at any degree. Shared cancelled factor J passes a complete 17-mode basis.")


def deficit_certificate():
    # b=2^128, r=b-7. Here j=r-k, M_j=2^(j+5) and Y_k/2^(k+1)
    # has a factor 1-4/2^(i+3) for every completed block with b_(i-1)>=j.
    # Truncating this product upward gives a LOWER bound on the optimized loss.
    ends = [i*(i+7)//2 for i in range(21)]
    deficit = Fraction(7)
    for j in range(1, 129):
        order = 1 << max(1, sum(end <= j for end in ends[1:]))
        q = 1 << max(1, (order.bit_length()-1)//2)
        alpha = 2*order
        modes = 1 << (j+5)
        length = modes-alpha
        # B(l+alpha)>=B(l), since alpha>=2q.
        assert alpha >= 2*q and length >= alpha
        best = max(endpoint_count(q, alpha, l) for l in range(length-alpha+1, length+1))
        retention = prod(1-Fraction(4, 1 << (i+3)) for i in range(1, 21) if ends[i-1] >= j)
        deficit += 1-retention*Fraction(best, 4*q*modes)
    assert deficit > Fraction(137, 4)
    assert deficit < Fraction(1503, 32)
    print("Certified lower bound on b-F(N)/N^2 from j=1..128 and at most 20 blocks:")
    print(deficit, "> 137/4")


def literature_slots():
    x = symbols("x")
    for n in (9, 27):
        base = chebyshevt(n, x)
        deriv = diff(base, x)
        odds = set(range(1, 2*n-2, 2))
        evens = set(range(2, 2*n-1, 2))
        hamiltonian = SymPoly(expand(base*deriv), x, domain=QQ)
        assert {i for (i,), coeff in hamiltonian.terms() if coeff} == set(range(1, 2*n, 2))
        for a in range(1, n-1, 2):
            mode = expand(deriv*chebyshevt(a, x))
            density = SymPoly(mode, x, domain=QQ)
            primitive = SymPoly(integrate(mode, x), x, domain=QQ)
            assert {i for (i,), c in density.terms() if c} <= odds
            assert {i for (i,), c in primitive.terms() if c} <= evens
        assert len(evens)*len(odds)+2*n == n*n+1
        print(f"Qin-Sun n={n}: degree cap {4*n-5}, at most {n*n+1} ordinary slots.")
    print("The slot count is an upper bound for the family. Mode coefficients are not assumed generic.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", action="store_true", help="only run checks for the article itself")
    options = parser.parse_args()
    quintic_transfer()
    deficit_certificate()
    if not options.core:
        literature_slots()
