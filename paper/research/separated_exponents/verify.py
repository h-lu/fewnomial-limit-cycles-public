"""Exact algebra, endpoint signs and polynomial expansion for separated exponents.

Run python3 -B -m research.separated_exponents.verify from the repository root.
"""

from argparse import ArgumentParser
from fractions import Fraction as R
from functools import cache
from math import comb, factorial, isqrt, lcm, prod

from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.saddle_moments.verify import envelope as paired_envelope, print_field
from research.varying_potential.verify import shape_system
from research.verify_term_bounds import benchmark, lift_and_collect, require


@cache
def entry(order, j):
    y = R(2 * j + 1, 2)
    return prod(y - i for i in range(order)) / (factorial(order) * y * (y - order))


@cache
def basis(degree):
    j, odd = divmod(degree, 2)
    matrix = [[entry(order, k) for k in range(j + odd, degree + 1)] for order in range(j + 1)]
    targets = [-entry(order, j) if odd else R(order == j) for order in range(j + 1)]
    inverse, _ = inverse_and_determinant(matrix)
    return tuple([R(0)] * j + [R(1)] * odd + matvec(inverse, targets))


def density(d, q):
    coefficients = [R(0)] * (d + 1)
    for i in range(d + 1):
        weight = (-1)**(i + (i + 1) // 2) * q**((d + 1 - i)**2 - 1)
        for power, a in enumerate(basis(i)):
            coefficients[power] += weight * a
    return coefficients


def interval_add_scaled(total, interval, factor):
    lower, upper = (factor * endpoint for endpoint in interval)
    if lower > upper:
        lower, upper = upper, lower
    return total[0] + lower, total[1] + upper


def sqrt_bounds(x, bits=384):
    scale = 2**bits
    root = isqrt((x.numerator * scale**2) // x.denominator)
    lo = R(root, scale)
    return lo, lo if lo**2 == x else R(root + 1, scale)


def center_mark(h, coefficients):
    # (1-s)^(-1) lies between 1 and 1/(1-h) throughout the slice.
    root = sqrt_bounds(h)
    total = (0, 0)
    for j, c in enumerate(coefficients):
        factor = 4 * c * h**(j + 1) / ((2 * j + 1) * (2 * j + 3))
        total = interval_add_scaled(total, (root[0], root[1] / (1 - h)), factor)
    return total


def upper_mark(r, coefficients):
    # The two coefficients of sqrt(h) and r arccos(r) are rational.
    h = 1 - r**2
    regular = singular = 0
    for j, c in enumerate(coefficients):
        scale = 2 * c / (2 * j + 1)
        regular += scale * sum((-r**2)**i * h**(j - i) * R(2, 2 * (j - i) + 1)
                               for i in range(j + 1))
        singular += 2 * scale * (-1)**(j + 1) * r**(2 * j + 1)
    # 3 < pi < 22/7 and r <= arcsin(r) <= 2r here.
    require(0 < r <= R(1, 2), "Upper mark inside the inverse-sine bound")
    total = interval_add_scaled((0, 0), sqrt_bounds(h), regular)
    return interval_add_scaled(total, (R(3, 2) - 2 * r, R(11, 7) - r), singular)


@cache
def vertical_system(p):
    full = shape_system(p)
    # Remove the horizontal tilt column and the final even Taylor row.
    minor = [row[:p] + row[p + 1:] for row in full.matrix[:-1]]
    inverse, determinant = inverse_and_determinant(minor)
    return minor, inverse, determinant


def expanded_field(k, p, delta, t, epsilon, coefficients, offset=None):
    if k < 1 or p < 2 or not 0 < delta < 1 or not 0 < t < 1 or epsilon <= 0:
        raise ValueError("Require k >= 1, p >= 2, 0 < delta,t < 1 and positive epsilon.")
    d = 3 * p - 1
    if len(coefficients) != d + 1:
        raise ValueError("Specify 3p rational coefficients in ascending order.")
    system = shape_system(p)
    _, inverse, _ = vertical_system(p)
    bound = 2 + max(sum(map(abs, row)) for row in inverse) * max(map(abs, coefficients))
    if offset is None:
        offset = int(bound) + 1
    if offset <= bound:
        raise ValueError("The given offset is too small for the coefficient norm.")
    controls = matvec(inverse, [c * t**(2 * (d - i)) for i, c in enumerate(coefficients)])
    weight_steps, power_steps = controls[:p] + [0], controls[p:2 * p]
    shape_steps, tau = controls[2 * p:-1] + [0], controls[-1]
    lam = t**(2 * d + 2)
    rho = lam * tau
    added = [lam * (step + offset * v) for step, v in zip(shape_steps, system.weights[1:])]
    amplitudes = [1 - sum(added)] + added
    if not all(amplitudes):
        raise ValueError("A coefficient vanishes at these parameters.")
    alpha = [epsilon * (v + lam * step) / amp
             for v, step, amp in zip(system.weights, weight_steps, amplitudes)]
    q = [node + lam * step / v for node, step, v in
         zip(system.nodes[1:], power_steps, system.weights[1:])]
    rates = [delta**i for i in range(1, k + 1)]
    scale = 2 * lcm(*(a.denominator for a in [*alpha, rho, *q, *rates]))
    first = [(amplitudes[0], alpha[0], 1 + rho), (-1, 0, 2 + rho)]
    first.extend((a, u, 1 + v) for a, u, v in zip(added, alpha[1:], q))
    second = [(t**2 * sum(rates) / 2, 1, 0)]
    second.extend((-t**2 * rate / 2, rate, 0) for rate in rates)
    P = lift_and_collect(first, scale, 1, 0)
    Q = lift_and_collect(second, scale, 0, 1)
    require((len(P), len(Q)) == (p + 2, k + 1), "Exact separately collected support")
    return P, Q


def eight_family(n):
    if n < 2:
        raise ValueError("Require an integer N >= 2.")
    t = R(1, n)
    coefficients = density(5, R(1, 16))
    return expanded_field(3, 2, R(1, 2**4096), t, t**100, coefficients, offset=10**6)


def check_eight():
    q = R(1, 16)
    coefficients = density(5, q)
    require(max(map(abs, coefficients)) + 1 < 32, "Uniform coefficient bound")
    require(sum(abs(c) * 3**i for i, c in enumerate(coefficients)) < 2**14,
            "Uniform bound for the density on [0,3]")
    values = [center_mark(q**(4 * (5 - 2 * j)), coefficients) for j in range(3)]
    values.extend(upper_mark(q**(2 * (6 - i)), coefficients) for i in reversed(range(6)))
    for i, endpoints in enumerate(values):
        oriented = [(-1)**i * value for value in endpoints]
        require(min(oriented) > R(1, 2**230), "Certified sign and absolute margin")
    delta = R(1, 2**4096)
    require(3 * 130 * delta + R(1, 2**128) < q**24, "Turning range at all nine marks")
    require(2**30 * R(1, 2**2048) < R(1, 2**231), "Finite exponent separation sign transfer")
    _, inverse, determinant = vertical_system(2)
    require(determinant == R(-1, 4762800) and max(sum(map(abs, r)) for r in inverse) == R(91804, 5),
            "Six vertical controls and their norm")
    require(tuple(map(len, eight_family(100))) == (4, 4), "One-integer support sample")
    print("Eight sign changes per interval at q=1/16, minimum certified margin > 2^-230.")
    print("All three finite intervals retain their signs at delta=2^-4096.")


def check_matrices():
    for a in range(10):
        for shift in (0, 1, 3):
            nodes = list(range(a + shift, 2 * a + shift + 1))
            M = [[entry(l, j) for j in nodes] for l in range(a + 1)]
            det = inverse_and_determinant(M)[1]
            # Undo the row and column factors in the alternant proof.
            for l in range(a + 1):
                det *= R(2**l * factorial(l), 4)
            Y = [2 * j + 1 for j in nodes]
            for y in Y:
                det *= y * prod(y - 2 * b for b in range(a + 1))
            vandermonde = prod(y - x for i, y in enumerate(Y) for x in Y[:i])
            quotient = det / vandermonde
            require(quotient.denominator == 1 and quotient.numerator % 2 == 1,
                    "Integral odd alternant quotient")
    for index in range(25):
        j, odd = divmod(index, 2)
        poly = basis(index)
        require(poly[-1] and not any(poly[:j]), "Degree and low coefficients")
        if odd:
            require(poly[j] == 1, "Normalized half-integer coefficient")
        for l in range(j + 1):
            require(sum(poly[k] * entry(l, k) for k in range(j, index + 1)) == (l == j and not odd),
                    "Endpoint conditions for the basis")
    for p in range(2, 10):
        minor, _, determinant = vertical_system(p)
        full = shape_system(p)
        # Independent cofactor from the full matrix with horizontal tilt.
        require(determinant == (-1)**(3 * p + p) * full.determinant * full.inverse[p][-1],
                "Deleting the last row and horizontal column gives the claimed minor")


def family_bound(m):
    return max(8 * (m - 3 - p) * ((9 * p - 2) // 4) for p in range(2, m - 3))


def print_table(last):
    print("m    Gasull-Santana   preceding hyperbolic   separated exponents")
    published = {6: 12, 7: 16, 8: 20, 9: 24, 10: 32}
    for m in range(6, last + 1):
        gs = max(benchmark(m)[0], published.get(m, 0))
        old = 12 * (m // 4) * (m - 2 - 2 * (m // 4))
        print(f"{m:2} {gs:15} {max(old, paired_envelope(m)):22} {family_bound(m):21}")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--field", type=int, metavar="N", help="emit the eight-term field at N")
    group.add_argument("--table", type=int, metavar="LAST")
    args = parser.parse_args()
    if args.field is not None:
        print_field(eight_family(args.field))
    elif args.table is not None:
        print_table(args.table)
    else:
        check_matrices()
        check_eight()
        for k, p in [(1, 2), (2, 3), (4, 4), (5, 3)]:
            t = R(1, 100)
            P, Q = expanded_field(k, p, R(1, 16), t, t**(8 * (3 * p - 1) + 9),
                                  density(3 * p - 1, R(1, 16)))
            require(all(x % 2 == 1 and y % 2 == 0 for x, y in P), "Parity in P")
            require(all(x % 2 == 0 and y % 2 == 1 for x, y in Q), "Parity in Q")
        print("Endpoint matrices, vertical minors and separate support checks passed.")
        print_table(16)


if __name__ == "__main__":
    main()
