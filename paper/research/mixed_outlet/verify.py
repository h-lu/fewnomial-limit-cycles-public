"""Exact model, vertical coefficients and polynomial support for the mixed outlet.

Run python3 -B -m research.mixed_outlet.verify from the repository root.
"""

import gzip
from argparse import ArgumentParser
from fractions import Fraction as Q
from functools import cache
from math import comb, factorial, lcm, prod
from pathlib import Path

from research.exponent_variations.verify import (
    binomial_derivative,
    inverse_and_determinant,
    matvec,
)
from research.separated_exponents.verify import basis, center_mark, density, upper_mark
from research.separated_exponents.witness import value_interval
from research.verify_term_bounds import require

K0 = Q(-15, 29)
SLOPE = Q(-76531, 180)
SELECTED = (0, 1, 2, 3, 4, 5, 7)
Q_MODEL = Q(1, 16)
T = DELTA = Q(1, 2**8192)
EPSILON_POWER = 1024


@cache
def coefficients():
    return tuple(a + b for a, b in zip(density(6, Q_MODEL) + [Q(0)], basis(7)))


def generalized_binomial(x, order):
    return prod(x - j for j in range(order)) * Q(1, factorial(order))


def raw_matrix(k, derivative=False):
    k = Q(k)
    v = 1 + 3 * k
    g = [-3 / v] + [k ** (i - 1) / v ** (i + 1) for i in range(1, 8)]
    dg = [9 / v**2]
    for i in range(1, 8):
        top = (i - 1) * k ** (i - 2) if i >= 2 else 0
        dg.append(top / v ** (i + 1) - 3 * (i + 1) * k ** (i - 1) / v ** (i + 2))
    base = (1 + 2 * k, -1, -3 * k, k)
    weights = (2, 0, -3, 1)
    shape = [Q(comb(2, 2 * j) - (j == 0)) for j in range(8)]
    logs = {s: [binomial_derivative(s, 2 * j) for j in range(8)] for s in (0, 2, 3)}

    def transport(order, series, factor):
        return (2 * order + 1) * sum(
            series[order - j] * factor[j] / (2 * j + 1) for j in range(order + 1)
        )

    rows = []
    for i in range(8):
        if derivative:
            row = [0, 0, transport(i, dg, shape), 0]
        else:
            row = [Q(i == 0), Q(comb(2, 2 * i)), transport(i, g, shape), Q(1)]
        for s, log in logs.items():
            if derivative:
                row.append(weights[s] * transport(i, g, log) + base[s] * transport(i, dg, log))
            else:
                row.append(weights[s] * log[i] + base[s] * transport(i, g, log))
        rows.append(row)
    return rows


def determinant(rows):
    try:
        return inverse_and_determinant(rows)[1]
    except ArithmeticError:
        return Q(0)


def check_rank():
    raw, direction = raw_matrix(K0), raw_matrix(K0, True)
    first = raw[:7]
    derivative = sum(determinant(first[:i] + [direction[i]] + first[i + 1 :]) for i in range(7))
    minor = determinant([raw[i] for i in SELECTED])
    require(
        determinant(first) == 0 and derivative == Q(-17249876309, 1648072065024000),
        "Simple determinant zero",
    )
    require(
        minor == Q(20511149, 833191988428800) and derivative / minor == SLOPE,
        "Seventh-degree direction and transverse slope",
    )
    # Clearing the row denominators leaves determinant polynomials of degrees <=28 and <=29.
    for k in map(Q, range(30)):
        rows = raw_matrix(k)
        require(
            determinant(rows[:7]) == -(29 * k + 15) / (98232750 * (3 * k + 1) ** 6),
            "First polynomial determinant identity",
        )
        require(
            determinant([rows[i] for i in SELECTED])
            == -(1319 * k * k + 1053 * k + 192) / (595945350 * (3 * k + 1) ** 7),
            "Second polynomial determinant identity",
        )


def energy_matrix(k):
    rows = raw_matrix(k)
    xi = -k / (2 * (1 + 3 * k))
    return [
        [
            sum(
                generalized_binomial(Q(-2 * i - 1, 2), i - j) * xi ** (i - j) * rows[j][c]
                for j in range(i + 1)
            )
            for c in range(7)
        ]
        for i in range(8)
    ]


def check_by_residue(k):
    """Differentiate [z^r] D(z) (2V(z)/(a*z^2))^(-(r+1)/2) at zero shape amplitude."""
    a = -(1 + 3 * k)
    xi = k / (2 * a)
    amplitudes = [1 + 2 * k, -1, -3 * k, k]
    gamma = [2, 0, -3, 1]
    matrix = energy_matrix(k)
    for col in range(7):
        force, divergence = [], []
        for j in range(16):
            if col == 0:
                f, s = 0, Q(j == 0)
            elif col == 1:
                f, s = 0, Q(comb(2, j))
            elif col == 2:
                f, s = comb(2, j) - (j == 0), 0
            elif col == 3:
                f, s = 0, (-1) ** j
            else:
                node = (0, 2, 3)[col - 4]
                log = sum(
                    Q((-1) ** (j - l + 1) * comb(node, l), j - l) for l in range(min(node + 1, j))
                )
                f, s = amplitudes[node] * log, gamma[node] * log
            force.append(f)
            divergence.append(s)
        for i in range(8):
            exponent = Q(-2 * i - 1, 2)
            result = sum(
                generalized_binomial(exponent, l) * xi**l * divergence[2 * (i - l)]
                for l in range(i + 1)
            )
            # D0=z^3-3z. Differentiate the power of the energy-coordinate quotient.
            for l in range(i + 1):
                for degree, c in ((1, -3), (3, 1)):
                    index = 2 * i - 2 * l - degree + 1
                    if index >= 0:
                        result += (
                            2
                            * exponent
                            / a
                            * generalized_binomial(exponent - 1, l)
                            * xi**l
                            * c
                            * Q(force[index], index + 1)
                        )
            require(result == matrix[i][col], "Independent energy-residue Taylor coefficient")


def polynomial_product(a, b):
    product = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] += x * y
    return product


def interpolate(values):
    """Power coefficients of a polynomial from its values at 0,...,r."""
    result = [Q(0)] * len(values)
    falling = [Q(1)]
    for order in range(len(values)):
        weight = values[0] / factorial(order)
        for j, c in enumerate(falling):
            result[j] += weight * c
        values = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        falling = polynomial_product(falling, [-order, 1])
    return result


@cache
def controls():
    """Seven Taylor coefficients plus the constant term in tau=t^2, for each control."""
    target = coefficients()
    kappa = target[6] / (target[7] * SLOPE)
    require(abs(kappa) < 1, "Bound on K slope")
    samples = [energy_matrix(Q(j)) for j in range(9)]
    entries = []
    rhs = []
    for i in range(8):
        row = []
        for c in range(7):
            vals = [(1 + 3 * k) ** (i + 1) * samples[k][i][c] for k in range(i + 2)]
            poly = interpolate(vals)
            # Substitute K=K0+kappa*tau in the degree at most i+1 row polynomial.
            row.append(
                [
                    kappa**j * sum(comb(l, j) * K0 ** (l - j) * poly[l] for l in range(j, len(poly)))
                    for j in range(len(poly))
                ]
            )
        entries.append(row)
        rhs.append(
            [Q(0)] * (7 - i)
            + [
                target[i] * comb(i + 1, j) * (1 + 3 * K0) ** (i + 1 - j) * (3 * kappa) ** j
                for j in range(i + 2)
            ]
        )
    # Read back the interpolants away from the interpolation nodes.
    test = Q(1, 8)
    k_test = K0 + kappa * test
    direct = energy_matrix(k_test)
    for i, row in enumerate(entries):
        require(
            all(
                sum(c * test**j for j, c in enumerate(poly)) == (1 + 3 * k_test) ** (i + 1) * x
                for poly, x in zip(row, direct[i])
            ),
            "Row interpolation",
        )
    inverse, _ = inverse_and_determinant([[p[0] for p in entries[i]] for i in SELECTED])
    steps = []
    for order in range(8):
        vector = []
        for i in SELECTED:
            row = entries[i]
            earlier = sum(
                steps[order - j][c] * poly[j]
                for c, poly in enumerate(row)
                for j in range(1, min(order + 1, len(poly)))
            )
            vector.append(rhs[i][order] - earlier)
        steps.append(matvec(inverse, vector))
    max_residual = Q(0)
    for i, row in enumerate(entries):
        residual = [Q(0)] * 16
        for c, poly in enumerate(row):
            product = polynomial_product(poly, [step[c] for step in steps])
            for j, value in enumerate(product):
                residual[j] += value
        for j, value in enumerate(rhs[i]):
            residual[j] -= value
        require(not any(residual[: 2 if i == 6 else 8]), "Vanishing normalized residual jets")
        max_residual = max(max_residual, sum(map(abs, residual)))
    control_norm = max(sum(abs(step[c]) for step in steps) for c in range(7))
    require(control_norm < 2**60 and max_residual < 2**60, "Finite coefficient and residual bounds")
    return kappa, steps, control_norm, max_residual


def check_marks():
    q, c = Q_MODEL, coefficients()
    require(max(map(abs, c)) < 128, "Density coefficients")
    lower = [q**24, q**16, q**8, Q(1, 8)]
    exponential = [center_mark(h, c) for h in lower[:3]] + [value_interval(lower[3], c)]
    exponential += [upper_mark(q ** (2 * (7 - i)), c) for i in reversed(range(7))]
    require(
        all(min((-1) ** i * x for x in bounds) > Q(1, 2**290) for i, bounds in enumerate(exponential)),
        "Ten model sign changes on each exponential range",
    )
    linear = [
        sum(4 * a * h**j / ((2 * j + 1) * (2 * j + 3)) for j, a in enumerate(c))
        for h in lower + [Q(2), Q(16)]
    ]
    require(all((-1) ** i * x > 0 for i, x in enumerate(linear)), "Five linear-range sign changes")


def parameter_values(t):
    kappa, steps, _, _ = controls()
    tau, lam = t * t, t**16
    x = [sum(step[c] * tau**j for j, step in enumerate(steps)) for c in range(7)]
    a0, a2, sigma, b, eta0, eta2, eta3 = x
    k = K0 + kappa * tau
    scale = -(1 + 3 * k)
    amplitudes = [1 + 2 * k - lam * sigma, Q(-1), -3 * k + lam * sigma, k]
    powers = [lam * eta0, Q(1), 2 + lam * eta2, 3 + lam * eta3]
    gamma = [2 + lam * a0, Q(0), -3 + lam * a2, Q(1)]
    return scale, amplitudes, powers, gamma, b


def split_binary(value):
    """Store a rational as an odd fraction times a signed power of two."""
    if not value:
        return Q(0), 0
    n, d = value.numerator, value.denominator
    n_shift = (n & -n).bit_length() - 1
    d_shift = (d & -d).bit_length() - 1
    return Q(n >> n_shift, d >> d_shift), n_shift - d_shift


def collect(terms):
    result = {}
    for x, y, c in terms:
        require(x >= 0 and y >= 0, "Nonnegative polynomial exponents")
        result[x, y] = result.get((x, y), 0) + c
    return {pair: c for pair, c in result.items() if c}


def expanded_field(t, delta, epsilon):
    scale, amps, powers, gamma, b = parameter_values(t)
    rates = [delta**i for i in (1, 2, 3)]
    g0 = scale * t * t * rates[-1] / 2
    # Separate the very long common power of two before the LCM and exact divisions.
    eps_odd, eps_shift = split_binary(epsilon)

    def perturbed(value):
        odd, shift = split_binary(value)
        return (odd * eps_odd, shift + eps_shift) if odd else (Q(0), 0)

    tilts = [perturbed(g / a) for g, a in zip(gamma, amps)]
    beta = perturbed(t**16 * b / g0)
    vertical = [split_binary(s) for s in powers]
    horizontal = [split_binary(r) for r in rates]
    all_powers = [*tilts, *vertical, *horizontal, beta]
    odd_lcm = lcm(*(odd.denominator for odd, _ in all_powers))
    binary_shift = 1 + max(0, *(-shift for _, shift in all_powers))
    denom = odd_lcm << binary_shift

    def scaled(power):
        odd, shift = power
        return ((odd_lcm // odd.denominator) * odd.numerator) << (binary_shift + shift)

    P = collect(
        (1 + denom + scaled(g), 2 * denom + scaled(s), c) for c, g, s in zip(amps, tilts, vertical)
    )
    terms = [
        (2 * denom, 1 + denom, -scale * t * t * sum(rates) / 2),
        (denom, 1 + denom + scaled(beta), g0),
    ]
    terms.extend(
        (denom + scaled(power), 1 + denom, scale * t * t * r / 2)
        for r, power in zip(rates[:2], horizontal[:2])
    )
    second = collect(terms)
    require((len(P), len(second)) == (4, 4), "Four reduced monomials in each component")
    return P, second


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--field", action="store_true", help="expand the fixed field and compare its archive"
    )
    group.add_argument(
        "--export",
        type=Path,
        metavar="FILE",
        help="write and check the full field in gzipped hexadecimal",
    )
    args = parser.parse_args()
    check_rank()
    for k in (K0, Q(-5, 9)):
        check_by_residue(k)
    check_marks()
    _, _, norm, res = controls()
    print("Transverse slope -76531/180; ten exponential and five linear sign changes.")
    print(f"Control coefficient norm < {int(norm) + 1}; row residual norm < {int(res) + 1}.")
    if args.field or args.export:
        from research.separated_exponents.witness import fingerprint

        result = expanded_field(T, DELTA, T**EPSILON_POWER)
        archive = args.export or Path(__file__).with_name("field.hex.gz")
        if args.export:
            with (
                archive.open("wb") as stream,
                gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as out,
            ):
                for name, component in zip(("P", "Q"), result):
                    out.write((name + "\n").encode("ascii"))
                    for (x, y), c in sorted(component.items()):
                        out.write(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}\n".encode("ascii"))
        with gzip.open(archive, "rt", encoding="ascii") as stream:
            for name, component in zip(("P", "Q"), result):
                require(stream.readline().strip() == name, "Component heading")
                for (x, y), c in sorted(component.items()):
                    data = tuple(int(item, 16) for item in stream.readline().split())
                    require(
                        data == (x, y, c.numerator, c.denominator),
                        "Expanded exponent pair and coefficient",
                    )
            require(stream.read() == "", "End of field")
        digest = fingerprint(result)
        require(
            digest == "e8b6ac8d1d5263ea476c95823158fcebc4b1c41c5b13d36e82500ee8210e595d",
            "Binary fingerprint",
        )
        sizes = [j.bit_length() for component in result for pair in component for j in pair]
        print(
            "Exact supports",
            tuple(map(len, result)),
            "exponent bit lengths",
            min(sizes),
            max(sizes),
            "SHA-256",
            digest,
        )
    else:
        require(
            tuple(map(len, expanded_field(Q(1, 1024), Q(1, 512), Q(1, 1024) ** 60))) == (4, 4),
            "Collected sample expansion",
        )


if __name__ == "__main__":
    main()
