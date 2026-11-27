"""Reproduce the exponent-variation construction using exact rational arithmetic.

From the repository root use python3 -B -m research.exponent_variations.verify.
Add --field S P N to print the two expanded polynomial components.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb, factorial, lcm
import sys

from research.verify_term_bounds import (
    benchmark as previous_bounds,
    disk_moment,
    lift_and_collect,
    require,
    root_product,
)


def inverse_and_determinant(matrix):
    """Gauss–Jordan elimination over the rationals."""
    size = len(matrix)
    rows = [
        [Fraction(v) for v in row] + [Fraction(i == j) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    determinant = Fraction(1)
    for j in range(size):
        pivot_row = next((i for i in range(j, size) if rows[i][j]), None)
        if pivot_row is None:
            raise ArithmeticError("Singular moment matrix")
        if pivot_row != j:
            rows[j], rows[pivot_row] = rows[pivot_row], rows[j]
            determinant = -determinant
        pivot = rows[j][j]
        determinant *= pivot
        rows[j] = [v / pivot for v in rows[j]]
        for i, row in enumerate(rows):
            if i != j and row[j]:
                factor = row[j]
                rows[i] = [v - factor * w for v, w in zip(row, rows[j])]
    return [row[size:] for row in rows], determinant


def binomial_derivative(node, degree):
    """Differentiate binom(a,degree) by its product recurrence, at a=node."""
    value, slope = Fraction(1), Fraction(0)
    for k in range(1, degree + 1):
        factor = node - k + 1
        slope = (slope * factor + value) / k
        value = value * factor / k
    return slope


def matvec(rows, vector):
    return [sum(v * w for v, w in zip(row, vector)) for row in rows]


@dataclass
class Moments:
    nodes: list
    baseline: list
    matrix: list
    inverse: list
    determinant: Fraction
    target: list


@cache
def moment_system(p):
    nodes = [2 * j for j in range(p)] + [2 * p - 1]
    # The top node has coefficient one. Solve for the cancelling polynomial
    # directly, independently of the tanh generating-function proof.
    baseline = [0] * p + [1]
    for k in reversed(range(p)):
        baseline[k] = -sum(baseline[j] * comb(nodes[j], 2 * k)
                           for j in range(k + 1, p + 1))
    d = 2 * p
    matrix = []
    for k in range(d + 1):
        amplitudes = [comb(node, 2 * k) for node in nodes[:-1]]
        slopes = [binomial_derivative(node, 2 * k) for node in nodes[1:]]
        matrix.append(amplitudes + [1] + slopes)
    inverse, determinant = inverse_and_determinant(matrix)
    target = [v * disk_moment(d) / disk_moment(k)
              for k, v in enumerate(root_product(d))]
    return Moments(nodes, baseline, matrix, inverse, determinant, target)


@dataclass
class Parameters:
    horizontal: list
    amplitudes: list
    powers: list
    epsilon: Fraction
    alpha: Fraction
    beta: Fraction
    scale: int


def parameters(s, p, n):
    if min(s, p, n) < 1:
        raise ValueError("s, p and N must be positive integers")
    t = Fraction(1, n)
    d = 2 * p
    lam, epsilon = t**(2 * d + 2), t**(4 * d + 5)
    moments = moment_system(p)
    rhs = [v * t**(2 * (d - k)) for k, v in enumerate(moments.target)]
    controls = matvec(moments.inverse, rhs)
    amplitudes = [v + lam * a for v, a in zip(moments.baseline, controls[:p] + [0])]
    powers = [node + lam * h / v
              for node, v, h in zip(moments.nodes[1:], moments.baseline[1:], controls[p + 1:])]
    horizontal = root_product(2 * s - 1)
    alpha = epsilon * amplitudes[0]
    beta = epsilon * lam * controls[p] / horizontal[0]
    if (min(alpha, beta) < -1 or min(powers) <= 0 or len(set(powers)) != p
            or beta in [1 + power for power in powers] or not all(amplitudes[1:])):
        raise ValueError("This N is outside the polynomial support conditions")
    scale = 2 * lcm(*(number.denominator for number in [alpha, beta] + powers))
    return Parameters(horizontal, amplitudes, powers, epsilon, alpha, beta, scale)


def expanded_field(s, p, n):
    data = parameters(s, p, n)
    first = [(1, data.alpha, 1), (-1, 0, 2)]
    second = [(data.horizontal[0], 0, data.beta)]
    second.extend((coefficient, j, 0) for j, coefficient in enumerate(data.horizontal) if j)
    for amplitude, power in zip(data.amplitudes[1:], data.powers):
        second.append((data.epsilon * amplitude / (1 + power), 0, 1 + power))
    first = lift_and_collect(first, data.scale, 1, 0)
    second = lift_and_collect(second, data.scale, 0, 1)
    require((len(first), len(second)) == (2, 2 * s + p), "Collected component supports")
    return first, second


def check_moment_system(p):
    d = 2 * p
    moments = moment_system(p)
    require(all(moments.baseline), "Every cancelling amplitude is nonzero")
    for k in range(p + 1):
        coefficient = sum(v * comb(node, 2 * k)
                          for node, v in zip(moments.nodes, moments.baseline))
        require(coefficient == 0, "Exact even cancellation")
    # Check the factorial reduction independently of the product derivative.
    for k in range(p, d + 1):
        for column, node in enumerate(moments.nodes[1:], p + 1):
            expected = (-1)**(node + 1) * factorial(node) * factorial(2 * k - 1 - node)
            require(factorial(2 * k) * moments.matrix[k][column] == expected,
                    "High-row factorial identity")
    for k in range(d + 1):
        column = [row[k] for row in moments.inverse]
        require(matvec(moments.matrix, column) == [int(i == k) for i in range(d + 1)],
                "Exact inverse of the jet matrix")
    # Disk moments by integration recurrence, independently of the binomial formula.
    moment = Fraction(1)
    averaged = []
    for k, coefficient in enumerate(moments.target):
        averaged.append(moment * coefficient)
        moment *= Fraction(2 * k + 1, 2 * k + 4)
    for j in range(1, d + 1):
        value = sum(a * j**k for k, a in enumerate(averaged))
        derivative = sum(k * averaged[k] * j**(k - 1) for k in range(1, d + 1))
        require(value == 0 and derivative != 0, "Simple prescribed disk-moment root")


def check_witness_field():
    n = 10**30
    t = Fraction(1, n)
    lam, epsilon = t**10, t**21
    # Integer multiple of the displayed 5 by 5 inverse, in column order
    # (A0, A2, R, L2, L3). This route starts from the specialized field.
    inverse_times_41 = [
        [41, -41, 1170, -14190, 13020],
        [0, 41, -1171, 14220, -13090],
        [0, 0, 1, -30, 70],
        [0, 0, 330, -6210, 5880],
        [0, 0, 270, -1950, 1680],
    ]
    rhs = [21 * t**8 / 16, -175 * t**6 / 16, 245 * t**4 / 16, -7 * t**2, 1]
    a0, a1, b, h1, h2 = [Fraction(v, 41) for v in matvec(inverse_times_41, rhs)]
    alpha = epsilon * (2 + lam * a0)
    beta = -epsilon * lam * b / 6
    q1, q2 = 2 - lam * h1 / 3, 3 + lam * h2
    scale = 2 * lcm(*(v.denominator for v in (alpha, beta, q1, q2)))
    p_field = {(1 + scale * (1 + alpha), 2 * scale): 1, (1 + scale, 3 * scale): -1}
    q_field = {
        (scale, 1 + scale * (1 + beta)): -6,
        (2 * scale, 1 + scale): 11,
        (3 * scale, 1 + scale): -6,
        (4 * scale, 1 + scale): 1,
        (scale, 1 + scale * (2 + q1)): epsilon * (-3 + lam * a1) / (1 + q1),
        (scale, 1 + scale * (2 + q2)): epsilon / (1 + q2),
    }
    require((p_field, q_field) == expanded_field(2, 2, n), "Eight-term witness expansion")
    system = moment_system(2)
    require(system.determinant == Fraction(41, 25200), "Witness determinant")
    require([[41 * v for v in row] for row in system.inverse] == inverse_times_41,
            "Independent witness inverse")
    require(max(sum(abs(v) for v in row) for row in system.inverse) < 800
            and max(map(abs, rhs)) < 16, "Witness control norm")

    # Compute the first five even coefficients with the actual shifted
    # rational exponents. The remainder bound on |z| <= 1/2 also gives a
    # Cauchy bound for each coefficient of the nonlinear remainder.
    power1 = power2 = Fraction(1)
    for order in range(9):
        if order % 2 == 0:
            constant = 2 + lam * a0 if order == 0 else 0
            actual = constant + lam * b + (-3 + lam * a1) * power1 + power2
            require(abs(actual - lam * rhs[order // 2]) <= 10**11 * lam**2 * 2**order,
                    "Exact coefficient at the actual shifted exponents")
        power1 *= (q1 - order) / (order + 1)
        power2 *= (q2 - order) / (order + 1)


def check_witness_bounds():
    t = Fraction(1, 10**30)
    lam, sigma, sigma_radius = t**10, t**20, Fraction(1, 10**6)
    h_bound = 20000
    # e^x <= 1/(1-x) for 0 <= x < 1; sqrt(2)>7/5 and sqrt(6)<49/20.
    potential_bound = sum(Fraction(c) / (1 - Fraction(k, 1000))
                          for c, k in [(243, 3), (216, 2), (33, 1)])
    require(potential_bound < 500, "Horizontal potential derivative")
    require(Fraction(1, 6) / Fraction(7, 5) < Fraction(12, 100) < Fraction(7, 5) - 1,
            "Inverse-coordinate disk")
    require(2 / (1 - Fraction(1, 1000))**2 < 3, "Inverse-coordinate derivative tail")
    require(Fraction(20, 49) - Fraction(3, 1000) > Fraction(2, 5)
            and Fraction(5, 7) + Fraction(3, 1000) < 1, "Jacobian range")
    require(800 * 16 < h_bound and (4 + 4 + 2 + 4 + 4) * h_bound < 10**6,
            "Analytic bound on the linearized density")
    require(2 * 10**6 * 6**10 < 2 * 10**14 and (6 * t)**2 < Fraction(1, 2),
            "Even Taylor tail")
    require(lam * h_bound < Fraction(1, 2) and 64 * h_bound**2 < 10**11,
            "Remainder from exponent variations")
    c = moment_system(2).target
    require(sum(abs(v) * 5**k for k, v in enumerate(c)) < 2500, "Disk polynomial bound")
    require(20 * (9000 * 2500 + 3 * 10**14 * t) < 10**9, "Energy variation error")
    require(3 * t <= Fraction(1, 10**6) and 3 * t / (1 - 3 * t) < 4 * t,
            "Inverse coordinate and logarithm domains")
    require(20 * t < Fraction(1, 2) and 6 * t * sigma_radius < Fraction(1, 2),
            "Angular exponential domains")
    angular_lower = 1 - 3 * t - 6 * (72 + 24) * sigma_radius
    require(angular_lower > Fraction(9, 10) and 6 * (72 + 24) / angular_lower < 1000,
            "Angular speed and radius variation")
    require(8 * 1000 * sigma_radius < Fraction(1, 4) and sigma < sigma_radius / 2,
            "One full turn and analytic remainder disk")
    higher_error = sigma / (sigma_radius**2 * lam * t**9)
    require(higher_error == 10**12 * t, "Normalized nonlinear error")
    lower = 3 * Fraction(2, 5) * disk_moment(4) * Fraction(3, 4)**4 / 4
    require(higher_error + 10**9 * t < Fraction(1, 300) < lower, "Four Rouché circles")


def run_checks():
    for p in range(1, 10):
        check_moment_system(p)
        for s in (1, 2, 4):
            expanded_field(s, p, 100)
    check_witness_field()
    check_witness_bounds()
    print("Exact cancellation, jet ranks, simple limiting roots and reduced supports passed.")
    print("The fixed field and the rational bounds for all four return disks passed.")
    theorem_two = {4: 12, 5: 12, 6: 12, 7: 16, 8: 20, 9: 24, 10: 32}
    print(" m   GS Thm.2   GS Prop.1   earlier C(m)   new F(m)")
    for m in range(5, 101):
        prop, old = previous_bounds(m)
        new = max(8 * s * (m - 2 - 2 * s) for s in range(1, (m - 3) // 2 + 1))
        remainder = (m - 2) % 4
        require(new == (m - 2)**2 - min(remainder, 4 - remainder)**2, "Integer envelope")
        require(new == 2 * prop, "Twice the optimized Proposition 1 count")
        if m >= 6:
            require(new > max(old, theorem_two.get(m, 0)), "Strict fixed-budget gain")
        if m <= 16:
            small = str(theorem_two.get(m, "-"))
            print(f"{m:2}  {small:>9}  {prop:10}  {old:13}  {new:9}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=int, nargs=3, metavar=("S", "P", "N"),
                        help="emit exact (x exponent, y exponent, coefficient) triples")
    args = parser.parse_args()
    if args.field is None:
        run_checks()
        return
    try:
        first, second = expanded_field(*args.field)
    except ValueError as error:
        parser.error(str(error))
    # Integer exponents in an explicitly requested expansion can exceed the
    # interpreter's default limit for decimal conversion.
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    for name, component in (("P", first), ("Q", second)):
        print(f"{name} has {len(component)} terms (x exponent, y exponent, coefficient)")
        for (x_power, y_power), coefficient in sorted(component.items()):
            print(x_power, y_power, coefficient)


if __name__ == "__main__":
    main()
