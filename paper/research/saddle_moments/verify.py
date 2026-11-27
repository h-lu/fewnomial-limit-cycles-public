"""Rational saddle data, exact support expansion, and finite-budget bounds.

Run python3 -B -m research.saddle_moments.verify from the repository root.
"""

from argparse import ArgumentParser
from fractions import Fraction
from functools import cache
from math import comb, factorial, lcm, prod
import sys

from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.varying_potential.verify import shape_system
from research.verify_term_bounds import benchmark, lift_and_collect, require, root_product


def odd_factorial(n):
    return prod(range(1, n + 1, 2))


@cache
def reduced_moment(j):
    return Fraction(4**(j + 1) * factorial(j), odd_factorial(4 * j + 3))


@cache
def entry(order, j):
    return reduced_moment(j - order) * Fraction(
        odd_factorial(2 * j - 1), 4**order * factorial(order)
    )


@cache
def basis(degree):
    """Coefficients of A_degree, in ascending powers with leading zeros retained."""
    j, odd = divmod(degree, 2)
    start = j + odd
    matrix = [[entry(order, power) for power in range(start, degree + 1)]
              for order in range(j + 1)]
    inverse, _ = inverse_and_determinant(matrix)
    targets = ([-entry(order, j) for order in range(j + 1)] if odd
               else [Fraction(order == j) for order in range(j + 1)])
    coefficients = [Fraction(0)] * j
    if odd:
        coefficients.append(Fraction(1))
    return tuple(coefficients + matvec(inverse, targets))


def count_changes(signs):
    return sum(left != right for left, right in zip(signs, signs[1:]))


def allocation(degree, start):
    """Inner count per well and full exterior count when starting at degree start."""
    inside = 2 * ((3 * start + 1) // 4)
    outside = start // 2 + (basis(start)[-1] < 0) + degree - start
    return inside, outside


@cache
def cluster_bound(degree):
    return max(2 * inner + outer for inner, outer in
               (allocation(degree, start) for start in range(1, degree + 1)))


def envelope(m):
    return max((4 * n * cluster_bound(3 * (m - 2 - 4 * n))
                for n in range(1, (m - 4) // 4 + 1)), default=0)


def multiply(first, second):
    result = [0] * (len(first) + len(second) - 1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            result[i + j] += a * b
    return result


def evaluate(poly, value):
    total = 0
    for coefficient in reversed(poly):
        total = total * value + coefficient
    return total


def horizontal(n, t, eta):
    """Coefficients g_1,...,g_(4n), expanding the shifted polynomial in U."""
    f = [1]
    for a in range(1, n + 1):
        f = multiply(f, [-1 - eta * a, 1])
    derivative = [i * f[i] for i in range(1, n + 1)]
    shifted_square = multiply(f, f)
    shifted_square[0] -= t**2 * eta**(2 * n)
    numerator = multiply(multiply(f, derivative), shifted_square)
    return [v / (t**2 * eta**(4 * n)) for v in numerator]


def expanded_field(n, p, t, coefficients, offset=None):
    """Collect both polynomial components at rational inputs satisfying support conditions."""
    if n < 1 or p < 2 or not 0 < t < 1 or len(coefficients) != 3 * p + 1:
        raise ValueError("Use n >= 1, p >= 2, 0 < t < 1, and 3p+1 coefficients.")
    d = 3 * p
    system = shape_system(p)
    weights = system.weights
    norm = max(sum(map(abs, row)) for row in system.inverse)
    bound = 2 + norm * max(map(abs, coefficients))
    if offset is None:
        offset = int(bound) + 1
    if offset <= bound:
        raise ValueError("The specified offset is too small for the density.")
    lam = t**(2 * d + 2)
    epsilon = t**(6 * d + 5)
    target = [c * t**(2 * (d - k)) for k, c in enumerate(coefficients)]
    controls = matvec(system.inverse, target)
    amplitude_steps = controls[:p] + [0]
    tilt = controls[p]
    exponent_steps = controls[p + 1:2 * p + 1]
    shape_steps = controls[2 * p + 1:-1] + [0]
    rho = lam * controls[-1]
    added = [lam * (s + offset * v) for s, v in zip(shape_steps, weights[1:])]
    amplitudes = [1 - sum(added)] + added
    if not all(amplitudes):
        raise ValueError("A zero amplitude at this t prevents the requested support.")
    alpha = [epsilon * (v + lam * a) / c
             for v, a, c in zip(weights, amplitude_steps, amplitudes)]
    q = [node + lam * step / v for node, step, v in
         zip(system.nodes[1:], exponent_steps, weights[1:])]
    g = horizontal(n, t, lam)
    if not all(g):
        raise ValueError("A horizontal coefficient vanishes at this t.")
    beta = epsilon * lam * tilt / g[0]
    scale = 2 * lcm(*(v.denominator for v in [*alpha, *q, rho, beta]))
    first = [(amplitudes[0], alpha[0], 1 + rho), (-1, 0, 2 + rho)]
    first.extend((c, a, 1 + power) for c, a, power in zip(added, alpha[1:], q))
    second = [(g[0], 1, beta)] + [(g[j], j + 1, 0) for j in range(1, len(g))]
    P = lift_and_collect(first, scale, 1, 0)
    Q = lift_and_collect(second, scale, 0, 1)
    require((len(P), len(Q)) == (p + 2, 4 * n), "Distinct nonzero support in P and Q")
    return P, Q


def check_moments():
    for j in range(33):
        # At the saddle use x = sqrt(2(1-y^2)); the remaining integral is a polynomial.
        independent = Fraction(2**(j + 2), 2 * j + 1) * sum(
            Fraction((-1)**k * comb(j, k), 2 * j + 2 * k + 3) for k in range(j + 1)
        )
        require(entry(0, j) == independent, "Independent saddle slice integral")
        for order in range(1, j + 1):
            require(order * entry(order, j) == Fraction(2 * j - 1, 4) * entry(order - 1, j - 1),
                    "Differentiated moment recursion")
    for j in range(12):
        for shift in (0, 1):
            pascal = [[Fraction(comb(shift + i + q, q)) for q in range(j + 1)]
                      for i in range(j + 1)]
            require(inverse_and_determinant(pascal)[1] == 1, "Shifted Pascal determinant")
            for i in range(j + 1):
                for q in range(j + 1):
                    u = shift + i + q
                    normalized = reduced_moment(u) / (4**(u + 1) * factorial(shift + i) * factorial(q))
                    require(normalized.denominator % 2 == 1 and
                            normalized.numerator % 2 == int(pascal[i][q]) % 2,
                            "Odd-denominator Hankel reduction modulo two")


def check_signs():
    for r in range(25):
        a = basis(r)
        j, odd = divmod(r, 2)
        require(a[-1] != 0 and not any(a[:j]), "Degree and lowest allowed power")
        if odd:
            require(a[j] == 1, "Normalized logarithmic coefficient")
        for order in range(j + 1):
            value = sum(a[k] * entry(order, k) for k in range(j, r + 1))
            require(value == int(not odd and order == j), "Defining saddle Taylor data")
        if r == 0:
            continue
        signed = [(-1)**(i + (i + 1) // 2) for i in range(r + 1)]
        centers = [signed[i] for i in range(1, r + 1, 2)]
        inner = [signed[i] * (-1)**((i + 1) // 2) for i in reversed(range(r + 1))]
        outer = signed + [signed[-1] * (1 if a[-1] > 0 else -1)]
        require((count_changes(centers + inner), count_changes(outer)) == allocation(r, r),
                "Two-sided saddle and center signs in physical energy order")
    expected = (0, 0, 1, Fraction(-24968385, 2558068),
                Fraction(4049469411, 143251808), Fraction(-6254809405, 245574528))
    require(basis(5) == expected and allocation(6, 5) == (8, 4),
            "Eight per well and four exterior for a sextic density")


def check_support():
    for n, p, denominator in [(1, 2, 100), (2, 2, 50), (3, 3, 80), (4, 4, 120)]:
        d = 3 * p
        t = Fraction(1, denominator)
        eta = t**(2 * d + 2)
        g = horizontal(n, t, eta)
        require(all((-1)**(j + 1) * g[j] > 0 for j in range(4 * n)), "All 4n coefficients")
        powers = root_product(n)
        derivative = [i * powers[i] for i in range(1, n + 1)]
        # Independently evaluate the factored formula at enough points to identify the polynomial.
        for i in range(4 * n + 1):
            X = Fraction(2 * i + 1, 3)
            U = 1 + eta * X
            f = evaluate(powers, X)
            expected = U * f * evaluate(derivative, X) * (f**2 - t**2) / (eta * t**2)
            require(U * evaluate(g, U) == expected, "Expanded horizontal polynomial identity")
        # This separate rational sample checks the lift and its support only.
        sample = [Fraction((-1)**k, k + 1) for k in range(d + 1)]
        first, second = expanded_field(n, p, t, sample)
        require(all(i % 2 == 1 and j % 2 == 0 for i, j in first), "Quadrant parity in P")
        require(all(i % 2 == 0 and j % 2 == 1 for i, j in second), "Quadrant parity in Q")


def print_table(last):
    print("m   GS Theorem 2/Prop 1   previous hyperbolic   cluster envelope")
    for m in range(8, last + 1):
        prior = 12 * (m // 4) * (m - 2 - 2 * (m // 4))
        gs = max(benchmark(m)[0], {8: 20, 9: 24, 10: 32}.get(m, 0))
        print(f"{m:<3} {gs:>12} {max(72, prior):>21} {envelope(m):>18}")


def print_field(components):
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    for name, component in zip(("P", "Q"), components):
        print(name)
        for (i, j), coefficient in sorted(component.items()):
            print(i, j, coefficient)


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--table", type=int, metavar="LAST")
    group.add_argument("--basis", type=int, metavar="DEGREE")
    group.add_argument("--field", nargs=4, metavar=("N", "P", "T", "COEFFICIENTS"),
                       help="ascending, comma-separated rational density coefficients")
    args = parser.parse_args()
    if args.table is not None:
        print_table(args.table)
    elif args.basis is not None:
        if args.basis < 0:
            parser.error("Specify a nonnegative basis degree.")
        print(", ".join(map(str, basis(args.basis))))
    elif args.field:
        n, p, t, coefficients = args.field
        print_field(expanded_field(int(n), int(p), Fraction(t),
                                   [Fraction(c) for c in coefficients.split(",")]))
    else:
        check_moments()
        check_signs()
        check_support()
        require(envelope(8) == 80, "Fixed eight-term improvement")
        print("Saddle integrals, parity determinants, defining Taylor systems and sign counts passed.")
        print("Exact horizontal expansion and both reduced polynomial support checks passed.")
        print_table(16)


if __name__ == "__main__":
    main()
