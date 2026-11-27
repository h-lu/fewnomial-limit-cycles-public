"""Exact coefficients, component supports and bounds for the varying-potential construction.

From the repository root use python3 -B -m research.varying_potential.verify.
Add --field S P N for the expanded polynomial at t=1/N.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb, factorial, lcm
import sys

from research.exponent_variations.verify import (
    binomial_derivative, inverse_and_determinant, matvec, moment_system,
)
from research.verify_term_bounds import disk_moment, lift_and_collect, require, root_product


def log_coefficient(order):
    return Fraction((-1)**(order + 1), order) if order else Fraction(0)


def product_coefficient(first, second, order):
    return sum(first[j] * second[order - j]
               for j in range(max(0, order - len(second) + 1), min(len(first), order + 1)))


@dataclass
class ShapeSystem:
    nodes: list
    weights: list
    matrix: list
    inverse: list
    determinant: Fraction
    target: list
    offset: int


@cache
def shape_system(p):
    if p < 2:
        raise ValueError("p must be at least two")
    previous = moment_system(p)
    nodes, weights = previous.nodes, previous.baseline
    d = 3 * p
    odd = [sum(v * comb(n, j) for v, n in zip(weights, nodes)) for j in range(2 * p)]
    require(not any(odd[::2]) and odd[-1] == 1, "Monic odd base density")
    # h_star/z has coefficient [z^k]log(1+z)/(k+2).
    shape_log = [log_coefficient(k) / (k + 2) for k in range(2 * d + 1)]
    # -h_n(z) is the primitive of ((1+z)^n-1), divided by z.
    shape_powers = [[0] + [Fraction(comb(n, k), k + 1) for k in range(1, n + 1)]
                    for n in nodes[1:]]
    matrix = []
    for k in range(d + 1):
        degree = 2 * k
        polynomial = [(degree + 1) * product_coefficient(odd, h, degree + 1)
                      for h in shape_powers]
        require(sum(v * col for v, col in zip(weights[1:], polynomial)) == 0,
                "Common offset vanishes in the first variation")
        log_shape = (weights[0] * log_coefficient(degree)
                     - (degree + 1) * product_coefficient(odd, shape_log, degree))
        row = [comb(n, degree) for n in nodes[:-1]] + [1]
        row += [binomial_derivative(n, degree) for n in nodes[1:]]
        matrix.append(row + polynomial[:-1] + [log_shape])
    inverse, determinant = inverse_and_determinant(matrix)
    target = [value * disk_moment(d) / disk_moment(k)
              for k, value in enumerate(root_product(d))]
    norm = max(sum(abs(value) for value in row) for row in inverse)
    offset = 1
    while offset <= 2 + norm * max(map(abs, target)):
        offset *= 10
    return ShapeSystem(nodes, weights, matrix, inverse, determinant, target, offset)


@dataclass
class Parameters:
    horizontal: list
    amplitudes: list
    alpha: list
    powers: list
    rho: Fraction
    beta: Fraction
    epsilon: Fraction
    scale: int


def parameters(s, p, n):
    if s < 1 or n < 1:
        raise ValueError("s and N must be positive")
    system = shape_system(p)
    d, t = 3 * p, Fraction(1, n)
    lam, epsilon = t**(2 * d + 2), t**(8 * d + 9)
    rhs = [v * t**(2 * (d - k)) for k, v in enumerate(system.target)]
    controls = matvec(system.inverse, rhs)
    a, b = controls[:p] + [0], controls[p]
    h = controls[p + 1:2 * p + 1]
    delta, rho = controls[2 * p + 1:-1] + [0], lam * controls[-1]
    weights = system.weights
    gamma = [v + lam * step for v, step in zip(weights, a)]
    powers = [node + lam * step / v for node, step, v in zip(system.nodes[1:], h, weights[1:])]
    added = [lam * (step + system.offset * v) for step, v in zip(delta, weights[1:])]
    amplitudes = [1 - sum(added)] + added
    if not all(amplitudes) or not all(gamma):
        raise ValueError("This N gives a zero amplitude")
    alpha = [epsilon * g / amplitude for g, amplitude in zip(gamma, amplitudes)]
    horizontal = root_product(2 * s - 1)
    beta = epsilon * lam * b / horizontal[0]
    vertical_exponents = [1 + rho, 2 + rho] + [1 + q for q in powers]
    if (len(set(vertical_exponents)) != p + 2
            or min(alpha + [beta] + vertical_exponents) < -1):
        raise ValueError("This N lies outside the support conditions")
    scale = 2 * lcm(*(v.denominator for v in alpha + [beta, rho] + powers))
    return Parameters(horizontal, amplitudes, alpha, powers, rho, beta, epsilon, scale)


def expanded_field(s, p, n):
    data = parameters(s, p, n)
    first = [(data.amplitudes[0], data.alpha[0], 1 + data.rho),
             (-1, 0, 2 + data.rho)]
    first.extend((c, alpha, 1 + q) for c, alpha, q in
                 zip(data.amplitudes[1:], data.alpha[1:], data.powers))
    second = [(data.horizontal[0], 0, data.beta)]
    second.extend((value, j, 0) for j, value in enumerate(data.horizontal) if j)
    first = lift_and_collect(first, data.scale, 1, 0)
    second = lift_and_collect(second, data.scale, 0, 1)
    require((len(first), len(second)) == (p + 2, 2 * s), "Separately collected supports")
    return first, second


def check_rank_and_roots(p):
    system = shape_system(p)
    d = 3 * p
    for j in range(1, p):
        # Polynomial shape for n=2j has degree 2p+2j-2.
        top = p + j - 1
        col = 2 * p + j
        require(system.matrix[top][col] == Fraction(2 * top + 1, 2 * j + 1)
                and all(row[col] == 0 for row in system.matrix[top + 1:]),
                "Triangular polynomial directions")
    # Calculate the missing log coefficients and the original factorial minors
    # separately from the power series used to construct the full matrix.
    k0 = 2 * p - 1
    fixed = [2 * k0] + [2 * k0 - 1 - n for n in system.nodes[1:]]
    signs = set()
    for j in range(p - 1):
        n = 2 * j + 1
        coefficient = -(j + 1) * sum(system.weights[:j + 1])
        indices = fixed + [2 * k0 - 1 - n]
        matrix = [[factorial(2 * i + q) for q in indices] for i in range(p + 2)]
        _, minor = inverse_and_determinant(matrix)
        signs.add((coefficient * minor > 0) - (coefficient * minor < 0))
    require(len(signs) == 1 and 0 not in signs, "Signed factorial minors add")
    require(system.determinant != 0, "Full column rank")
    moment = Fraction(1)
    averaged = []
    for k, value in enumerate(system.target):
        averaged.append(moment * value)
        moment *= Fraction(2 * k + 1, 2 * k + 4)
    for j in range(1, d + 1):
        value = sum(a * j**k for k, a in enumerate(averaged))
        slope = sum(k * averaged[k] * j**(k - 1) for k in range(1, d + 1))
        require(value == 0 and slope != 0, "Simple positive moment roots")


def check_transported_coefficients(t, rho, powers, amplitudes, gamma, b, target):
    """Recompute the moving density at the fixed parameter by change of residue."""
    d = len(target) - 1
    lam = t**(2 * d + 2)

    def binomial_series(exponent):
        values = [Fraction(1)]
        for j in range(1, 2 * d + 2):
            values.append(values[-1] * (exponent - j + 1) / j)
        return values

    common = binomial_series(rho)
    shifted = [binomial_series(q) for q in powers]
    potential, density = [], []
    for j in range(2 * d + 1):
        correction = sum(c * (row[j + 1] - common[j + 1])
                         for c, row in zip(amplitudes, shifted))
        potential.append(2 * (common[j] - correction) / (j + 2))
        density.append(gamma[0] * common[j] + lam * b * (-1)**j
                       + sum(g * row[j] for g, row in zip(gamma[1:], shifted)))
    # U=2V/z^2 and Y=z sqrt(U). Multiplication by sqrt(U(0)) makes
    # each even coefficient of the transported density rational.
    normalized = [value / potential[0] for value in potential]
    for k in range(d + 1):
        m = 2 * k
        exponent = Fraction(-m - 1, 2)
        powered = [Fraction(1)]
        # For B=A^exponent use A B' = exponent A' B, with A(0)=1.
        for degree in range(1, m + 1):
            powered.append(sum(((exponent + 1) * j - degree) * normalized[j] * powered[degree - j]
                               for j in range(1, degree + 1)) / degree)
        actual = product_coefficient(density, powered, m) / potential[0]**k
        linear = lam * target[k] * t**(2 * d - m)
        require(abs(actual - linear) <= 240 * (lam * 10**12)**2 * 4**m,
                "Nonlinear coefficient in the actual moving coordinate")


def check_fixed_field():
    # Independent of differentiation of the moving coordinate and logarithmic
    # powers above. Start with the specialized seven rows printed in the note.
    f = Fraction
    matrix = [
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, f(3, 2), f(5, 2), -3, 2],
        [0, 0, 1, f(-1, 12), f(1, 4), f(5, 3), f(-7, 6)],
        [0, 0, 1, f(-1, 60), f(1, 60), 0, f(-1, 5)],
        [0, 0, 1, f(-1, 168), f(1, 280), 0, f(-11, 140)],
        [0, 0, 1, f(-1, 360), f(1, 840), 0, f(-13, 315)],
        [0, 0, 1, f(-1, 660), f(1, 1980), 0, f(-5, 198)],
    ]
    target = [f(1485, 64), f(-14553, 64), f(6699, 16), f(-4851, 16),
              f(825, 8), f(-33, 2), 1]
    inverse, determinant = inverse_and_determinant(matrix)
    system = shape_system(2)
    require(matrix == system.matrix and system.target == target, "Specialized moment matrix")
    require(determinant == f(-1, 39293100)
            and max(sum(map(abs, row)) for row in inverse) == f(370714, 5),
            "Witness determinant and inverse norm")
    n = 10**50
    t, lam, eps = f(1, n), f(1, n**14), f(1, n**57)
    rhs = [v * t**(12 - 2 * k) for k, v in enumerate(target)]
    a0, a1, b, h1, h2, delta, tau = matvec(inverse, rhs)
    require(max(map(abs, (a0, a1, b, h1, h2, delta, tau))) < 4 * 10**7
            and system.offset == 10**8, "Witness parameter range")
    c1, c2 = lam * (delta - 3 * 10**8), lam * 10**8
    c0 = 1 - c1 - c2
    rho, beta = lam * tau, -eps * lam * b / 6
    q1, q2 = 2 - lam * h1 / 3, 3 + lam * h2
    alpha0, alpha1, alpha2 = eps * (2 + lam * a0) / c0, eps * (-3 + lam * a1) / c1, eps / c2
    scale = 2 * lcm(*(v.denominator for v in (rho, beta, q1, q2, alpha0, alpha1, alpha2)))
    first = {(1 + scale * (1 + alpha0), scale * (2 + rho)): c0,
             (1 + scale, scale * (3 + rho)): -1,
             (1 + scale * (1 + alpha1), scale * (2 + q1)): c1,
             (1 + scale * (1 + alpha2), scale * (2 + q2)): c2}
    second = {(scale, 1 + scale * (1 + beta)): -6, (2 * scale, 1 + scale): 11,
              (3 * scale, 1 + scale): -6, (4 * scale, 1 + scale): 1}
    require((first, second) == expanded_field(2, 2, n), "Independent eight-term field")
    check_transported_coefficients(t, rho, (q1, q2), (c1, c2),
                                   (2 + lam * a0, -3 + lam * a1, 1), b, target)


def check_fixed_estimates():
    f = Fraction
    t, r, sr = f(1, 10**50), f(1, 10**12), f(1, 10**6)
    lam, sigma = t**14, t**42
    target = shape_system(2).target
    require(f(370714, 5) * max(map(abs, target)) < 4 * 10**7, "Fixed control bound")
    require(4 * 10**7 * r < f(1, 10000) and (4 * 10**7 + 4 * 10**8) * r < f(5, 10000),
            "Complex shaping parameter disk")
    require(f(3, 2)**3 / (1 - 4 * 10**7 * r) < 4
            and 2 * 4 * 10**7 * r < f(1, 10000), "Exponential amplitude bounds")
    require(f(1, 10000) + 40 * f(5, 10000) < f(1, 32) and f(2, 31) < f(1, 10),
            "Vertical coordinate and derivative")
    require(2 * (3 * 2 + 5 * 4 + f(2, 10000)) < 60, "Holomorphic transported density")
    require(12 * t < f(1, 2) and lam < r / 2
            and 120 * (12**14 / r + 1 / r**2) < 10**30, "Scaled density error")
    horizontal_bound = sum(f(v) / (1 - f(j, 1000)) for v, j in [(243, 3), (216, 2), (33, 1)])
    require(horizontal_bound < 500 and f(1, 6) / f(7, 5) < f(12, 100)
            and f(2) / (1 - f(1, 1000))**2 < 3, "Horizontal inverse estimates")
    require(f(20, 49) - f(3, 1000) > f(2, 5) and f(5, 7) + f(3, 1000) < 1,
            "Horizontal Jacobian range")
    require(sum(abs(c) * 7**k for k, c in enumerate(target)) < 10**6
            and 28 * (9000 * 10**6 + 10**30 * t) < 10**12, "Energy integral error")
    require(f(32, 31) * 3 < 4 and 4 * t / (1 - 4 * t) < 5 * t, "Angular-coordinate domains")
    lower_speed = (1 - 4 * t) * f(9, 10) - 6 * (160 + 4) * sr
    require(lower_speed > f(4, 5) and 6 * (160 + 4) / lower_speed < 1300
            and 8 * 1300 * sr < f(1, 4), "Complex one-turn continuation")
    require(sigma < sr / 2 and sigma / (sr**2 * lam**2 * t**13) == 10**12 * t,
            "Full return Taylor error")
    lower = 3 * f(2, 5) * disk_moment(6) * f(3, 4)**6 / 4
    require(2 * 10**12 * t < f(1, 1000) < lower, "Six disjoint Rouché disks")


def run_checks():
    for p in range(2, 9):
        check_rank_and_roots(p)
        for s in (1, 2, 4):
            expanded_field(s, p, 1000)
    check_fixed_field()
    check_fixed_estimates()
    print("Ranks, prescribed simple roots, expanded supports and fixed-field bounds passed.")
    print(" m  previous  varying potential")
    for m in range(6, 201):
        old = max(8 * s * (m - 2 - 2 * s) for s in range(1, (m - 3) // 2 + 1))
        new = max(12 * s * (m - 2 - 2 * s) for s in range(1, (m - 4) // 2 + 1))
        s = m // 4
        require(2 * new == 3 * old and new == 12 * s * (m - 2 - 2 * s),
                "Exact fixed-support comparison")
        if m <= 16:
            print(f"{m:2}  {old:8}  {new:17}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=int, nargs=3, metavar=("S", "P", "N"),
                        help="print exact (x exponent, y exponent, coefficient) triples")
    args = parser.parse_args()
    if args.field is None:
        run_checks()
        return
    try:
        first, second = expanded_field(*args.field)
    except ValueError as error:
        parser.error(str(error))
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    for name, component in (("P", first), ("Q", second)):
        print(f"{name} has {len(component)} terms (x exponent, y exponent, coefficient)")
        for (x_power, y_power), value in sorted(component.items()):
            print(x_power, y_power, value)


if __name__ == "__main__":
    main()
