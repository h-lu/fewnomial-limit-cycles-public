"""Exact two-energy interpolation, vertical controls and the collected polynomial.

Run python3 -B -m research.tail_interpolation.verify from the repository root.
"""

import gzip
from argparse import ArgumentParser
from fractions import Fraction as Q
from functools import cache
from math import comb, lcm
from pathlib import Path

from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.mixed_outlet.verify import (
    DELTA, EPSILON_POWER, K0, SELECTED, SLOPE, T, collect, energy_matrix,
    interpolate, polynomial_product, split_binary,
)
from research.separated_exponents.verify import basis, density, entry
from research.separated_exponents.witness import fingerprint
from research.verify_term_bounds import require


Q_MODEL = Q(1, 16)
NODES = (Q(15, 16), Q(49, 50))
ARCHIVE = Path(__file__).with_name("field.hex.gz")


def value(coefficients, x):
    total = 0
    for coefficient in reversed(coefficients):
        total = total * x + coefficient
    return total


def tail_polynomial(c):
    return [4 * a / ((2 * j + 1) * (2 * j + 3)) for j, a in enumerate(c)]


@cache
def target():
    base = density(5, Q_MODEL) + [Q(0), Q(0)]
    columns = (basis(6) + (Q(0),), basis(7))
    endpoint_minor = [[entry(i, j) for j in (3, 4, 5)] for i in range(3)]
    require(inverse_and_determinant(endpoint_minor)[1] == Q(183616, 1444021425),
            "Exact dimension of the endpoint kernel")
    for column in columns:
        require(not any(column[:3]), "Three central density coefficients are fixed")
        for order in range(3):
            require(sum(entry(order, j) * a for j, a in enumerate(column)) == 0,
                    "Analytic endpoint coefficients through order two are fixed")
    matrix = [[value(tail_polynomial(c), h) for c in columns] for h in NODES]
    inverse, det = inverse_and_determinant(matrix)
    require(det == Q(32428882679001115293, 2883708677590614016000000000),
            "Two independent finite-energy evaluations")
    a, b = matvec(inverse, [-value(tail_polynomial(base), h) for h in NODES])
    c = tuple((z + a * x + b * y) / 32 for z, x, y in zip(base, *columns))
    tail = tail_polynomial(c)
    require(all(value(tail, h) == 0 for h in NODES), "Both prescribed roots are exact")
    require(c[-1] < 0 and max(map(abs, c)) < 48, "Degree seven and coefficient norm")
    return c, a, b


@cache
def controls():
    c, _, _ = target()
    kappa = c[6] / (SLOPE * c[7])
    samples = [energy_matrix(Q(k)) for k in range(9)]
    rows, targets = [], []
    for i in range(8):
        row = []
        for col in range(7):
            # Clear the common denominator before translating K to K0+kappa*tau.
            coefficients = interpolate([
                (1 + 3*k)**(i+1) * samples[k][i][col] for k in range(i+2)
            ])
            row.append([
                kappa**j * sum(comb(l, j) * K0**(l-j) * coefficients[l]
                               for l in range(j, len(coefficients)))
                for j in range(len(coefficients))
            ])
        rows.append(row)
        targets.append([Q(0)] * (7-i) + [
            c[i] * comb(i+1, j) * (1+3*K0)**(i+1-j) * (3*kappa)**j for j in range(i+2)
        ])
    crosscheck = Q(1, 8)
    test_k = K0 + kappa * crosscheck
    direct = energy_matrix(test_k)
    for i, row in enumerate(rows):
        require([value(poly, crosscheck) for poly in row]
                == [(1+3*test_k)**(i+1) * z for z in direct[i]], "Row interpolation readback")

    inverse, _ = inverse_and_determinant([[p[0] for p in rows[i]] for i in SELECTED])
    steps = []
    for order in range(8):
        rhs = []
        for i in SELECTED:
            previous = sum(steps[order-j][col] * p[j] for col, p in enumerate(rows[i])
                           for j in range(1, min(len(p), order+1)))
            rhs.append(targets[i][order] - previous)
        steps.append(matvec(inverse, rhs))

    residual_norm = Q(0)
    for i, row in enumerate(rows):
        residual = [Q(0)] * 16
        for col, p in enumerate(row):
            for j, z in enumerate(polynomial_product(p, [step[col] for step in steps])):
                residual[j] += z
        for j, z in enumerate(targets[i]):
            residual[j] -= z
        require(not any(residual[:2 if i == 6 else 8]), "Finite vertical residual orders")
        residual_norm = max(residual_norm, sum(map(abs, residual)))
    control_norm = max(sum(abs(step[j]) for step in steps) for j in range(7))
    require(abs(kappa) < 1 and max(residual_norm, control_norm) < 2**60,
            "Finite-scale hypotheses on the controls")
    return kappa, steps, control_norm, residual_norm


def expanded_field():
    kappa, steps, _, _ = controls()
    t, delta = T, DELTA
    lam = t**16
    a0, a2, sigma, b, eta0, eta2, eta3 = [
        value(column, t*t) for column in zip(*steps)
    ]
    k = K0 + kappa*t*t
    a = -(1+3*k)
    amps = [1+2*k-lam*sigma, Q(-1), -3*k+lam*sigma, k]
    powers = [lam*eta0, Q(1), 2+lam*eta2, 3+lam*eta3]
    weights = [2+lam*a0, Q(0), -3+lam*a2, Q(1)]
    rates = [delta, delta**2, delta**3]
    g0 = a*t*t*rates[-1]/2
    require(Q(1, 2) < a < 1 and all(Q(1, 40) < abs(v) < 4 for v in amps),
            "All four first-component coefficients survive")
    require(all(abs(v-j) < Q(1, 4) for j, v in enumerate(powers)), "Distinct vertical powers")
    require(max(abs(v/c) for v, c in zip(weights, amps)) < 256 and abs(lam*b/g0) < 1,
            "Exponent bounds at finite parameters")

    # Keep the common dyadic perturbation as a shift until after the LCM.
    require(t == Q(1, 2**8192), "Fixed dyadic vertical and perturbation scales")
    def perturbed(v):
        odd, shift = split_binary(v)
        return (odd, shift - 8192*EPSILON_POWER) if odd else (Q(0), 0)

    tilt = [perturbed(v/c) for v, c in zip(weights, amps)]
    beta = perturbed(lam*b/g0)
    nu = [split_binary(v) for v in powers]
    horizontal = [split_binary(r) for r in rates]
    rational_powers = [*tilt, *nu, *horizontal, beta]
    odd_lcm = lcm(*(odd.denominator for odd, _ in rational_powers))
    shift = 1 + max(0, *(-s for _, s in rational_powers))
    d = odd_lcm << shift

    def scale(power):
        odd, s = power
        return ((odd_lcm // odd.denominator) * odd.numerator) << (shift+s)

    p = collect((1+d+scale(u), 2*d+scale(v), c) for c, u, v in zip(amps, tilt, nu))
    q = collect([
        (2*d, 1+d, -a*t*t*sum(rates)/2),
        (d+scale(horizontal[0]), 1+d, a*t*t*rates[0]/2),
        (d+scale(horizontal[1]), 1+d, a*t*t*rates[1]/2),
        (d, 1+d+scale(beta), g0),
    ])
    require((len(p), len(q)) == (4, 4), "Separate expanded supports")
    require(all(x%2 == 1 and y%2 == 0 for x, y in p)
            and all(x%2 == 0 and y%2 == 1 for x, y in q), "Four-quadrant parity")
    return p, q


def check_field(export=False):
    components = expanded_field()
    if export:
        with ARCHIVE.open("wb") as stream, gzip.GzipFile(
                filename="", fileobj=stream, mode="wb", mtime=0) as out:
            for name, part in zip(("P", "Q"), components):
                out.write((name+"\n").encode("ascii"))
                for (x, y), c in sorted(part.items()):
                    out.write(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}\n".encode("ascii"))
    with gzip.open(ARCHIVE, "rt", encoding="ascii") as stream:
        for name, part in zip(("P", "Q"), components):
            require(stream.readline().strip() == name, "Archive component")
            for (x, y), c in sorted(part.items()):
                require(tuple(int(s, 16) for s in stream.readline().split())
                        == (x, y, c.numerator, c.denominator), "Expanded field readback")
        require(stream.read() == "", "End of archived polynomial")
    sizes = [v.bit_length() for part in components for pair in part for v in pair]
    digest = fingerprint(components)
    require(digest == "01fb565a9b281cd37b0e97a6218c4a120894bc65f1164f5f7fc275b7663d3b83",
            "Canonical binary fingerprint of the fixed polynomial")
    print("Collected (4, 4), exponent bits", min(sizes), max(sizes),
          "binary SHA-256", digest)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--field", action="store_true", help="expand and compare the archived polynomial")
    parser.add_argument("--export", action="store_true", help="regenerate the fixed archive and compare")
    args = parser.parse_args()
    _, a, b = target()
    _, _, cn, rn = controls()
    print("Tail roots fixed at", *NODES, "with interpolation parameters", a, b)
    print("Control norm <", int(cn)+1, "and residual norm <", int(rn)+1)
    if args.field or args.export:
        check_field(args.export)


if __name__ == "__main__":
    main()
