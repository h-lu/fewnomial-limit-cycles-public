"""Exact arithmetic for the polynomial construction in Section U2 of the unrestricted-degree attachment.

Run without arguments for the checks. Use --field S D N to print both
expanded components as (x exponent, y exponent, coefficient) triples.
Only the Python standard library is required.
"""

from argparse import ArgumentParser
from fractions import Fraction
from math import comb, factorial, lcm


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def root_product(count):
    """Coefficients of (X-1)...(X-count), in ascending powers."""
    values = [1]
    for root in range(1, count + 1):
        result = [0] * (len(values) + 1)
        for power, value in enumerate(values):
            result[power] -= root * value
            result[power + 1] += value
        values = result
    return values


def disk_moment(k):
    return Fraction(comb(2 * k, k), 4**k * (k + 1))


def choose_coefficients(d, t):
    c = [disk_moment(d) * value / disk_moment(k)
         for k, value in enumerate(root_product(d))]
    a = [Fraction(0)] * d
    for k in reversed(range(d)):
        higher = sum(comb(2 * ell, 2 * k) * a[ell] for ell in range(k + 1, d))
        a[k] = c[k] * t**(2 * (d - k)) - 1 - higher
    return c, a


def lift_and_collect(terms, scale, x_offset, y_offset):
    """Apply the log coordinate change and common time factor to one component."""
    expanded = {}
    for coefficient, u_power, v_power in terms:
        x_base = Fraction(scale * (1 + u_power))
        y_base = Fraction(scale * (1 + v_power))
        for base in (x_base, y_base):
            require(base.denominator == 1 and base >= 0 and base % 2 == 0,
                    "The lifted exponent must be a nonnegative even integer")
        key = (int(x_base) + x_offset, int(y_base) + y_offset)
        expanded[key] = expanded.get(key, 0) + coefficient
    return {key: value for key, value in expanded.items() if value}


def expanded_field(s, d, n):
    if min(s, d, n) < 1:
        raise ValueError("s, d and N must be positive integers")
    t = Fraction(1, n)
    epsilon = t**(2 * d + 3)
    b = root_product(2 * s - 1)
    c, a = choose_coefficients(d, t)
    scale = 2 * n**(4 * d + 3) * lcm(abs(b[0]), *(v.denominator for v in c))
    alpha = epsilon * a[0]
    beta = epsilon / b[0]
    if min(alpha, beta) < -1 or any(value == 0 for value in a[1:]):
        raise ValueError("This N is outside the polynomial support conditions")

    # These lists describe the exponentials in the analytic system itself.
    first = [(1, alpha, 1), (-1, 0, 2)]
    second = [(b[0], 0, beta)]
    second.extend((b[j], j, 0) for j in range(1, len(b)))
    for ell in range(1, d):
        power = 2 * ell + 1
        second.append((epsilon * a[ell] / power, 0, power))
    p = lift_and_collect(first, scale, 1, 0)
    q = lift_and_collect(second, scale, 0, 1)
    require((len(p), len(q)) == (2, 2 * s + d - 1), "Collected component supports")
    return p, q


def benchmark(m):
    proposition = max((2 * (m - r - 4) * (r + 1) + (m - r - 4) * (1 + (-1)**r)
                       for r in range(m - 4)), default=0)
    family = max(4 * s * (m - 1 - 2 * s) for s in range(1, (m - 2) // 2 + 1))
    return proposition, family


def check_moments(d, t):
    c, a = choose_coefficients(d, t)
    # Expand the even powers and compare both numerators after multiplying
    # the exact integrand by 1-X, where X=z^2.
    even = [sum(a[ell] * comb(2 * ell, 2 * k) for ell in range(k, d))
            for k in range(d)]
    target = [c[k] * t**(2 * (d - k)) for k in range(d)]
    numerator = [x - y for x, y in zip(even + [0], [0] + even)]
    target_numerator = [x - y for x, y in zip(target + [0], [0] + target)]
    numerator[0] += 1
    target_numerator[-1] += 1
    require(numerator == target_numerator, "Exact rational integrand identity")

    # Polar disk moments from their integral recurrence, independently of
    # the binomial expression used to set the coefficients.
    moment = Fraction(1)
    averaged = []
    for k, coefficient in enumerate(c):
        averaged.append(moment * coefficient)
        moment *= Fraction(2 * k + 1, 2 * k + 4)
    for root in range(1, d + 1):
        value = sum(term * root**k for k, term in enumerate(averaged))
        slope = sum(k * averaged[k] * root**(k - 1) for k in range(1, d + 1))
        expected = disk_moment(d) * (-1)**(d - root) * factorial(root - 1) * factorial(d - root)
        require(value == 0 and slope == expected != 0, "Simple averaged root")


def check_specified_field(n):
    t = Fraction(1, n)
    epsilon = t**9
    scale = 192 * n**15
    shift_x = -6 * (160 * n**6 + 600 * n**4 + 110 * n**2 + 15)
    shift_y = -32 * n**6
    a1 = 5 + Fraction(45, 2) * t**2 + Fraction(55, 16) * t**4
    a2 = -1 - Fraction(15, 4) * t**2
    p = {(1 + scale + shift_x, 2 * scale): 1, (1 + scale, 3 * scale): -1}
    q = {
        (scale, 1 + scale + shift_y): -6,
        (2 * scale, 1 + scale): 11,
        (3 * scale, 1 + scale): -6,
        (4 * scale, 1 + scale): 1,
        (scale, 1 + 4 * scale): epsilon * a1 / 3,
        (scale, 1 + 6 * scale): epsilon * a2 / 5,
    }
    require((p, q) == expanded_field(2, 3, n), "Eight-term displayed field")


def check_error_bounds():
    """Check scalar inequalities at the sample scale. U2 proves the return limit as t tends to zero."""
    t = Fraction(1, 10**20)
    sigma_radius = Fraction(1, 10**6)
    c, a = choose_coefficients(3, t)
    # e^x <= 1/(1-x) on 0 <= x < 1.
    second_derivative = sum(Fraction(v, 1) / (1 - Fraction(k, 1000))
                            for v, k in [(243, 3), (216, 2), (33, 1)])
    require(second_derivative < 500, "Bound on the horizontal potential derivative")
    sqrt_error = Fraction(1, 6) / Fraction(7, 5)
    require(sqrt_error < Fraction(12, 100) < Fraction(7, 5) - 1, "Inverse disk")
    require(2 / (1 - Fraction(1, 1000))**2 < 3, "Cauchy bound on K variation")
    require(Fraction(20, 49) - Fraction(3, 1000) > Fraction(2, 5)
            and Fraction(5, 7) + Fraction(3, 1000) < 1, "Bounds on the Jacobian")
    bracket = sum(abs(c[k]) * 4**k for k in range(3)) + 4**3 / (1 - 4 * t**2)
    require(bracket < 150 and 4**4 / (1 - 4 * t**2) < 512, "Disk integrand bounds")
    require(16 * (6000 * 150 + 512 * t) < 2 * 10**7, "First-variation error")

    require(abs(a[0]) < 6 and abs(a[1]) < 6 and abs(a[2]) < 2, "Drift coefficients")
    require(2 * t / (1 - 2 * t) < 1 and (1 + 2 * t)**5 < 2, "Log and power bounds")
    require(12 * t * sigma_radius < Fraction(1, 2), "Exponential perturbation")
    require(2 * 3 * 2 * 6 * 2 == 144, "First velocity bound")
    require(2 * (2 + 2 * (Fraction(6, 3) + Fraction(2, 5))) < 14,
            "Second velocity bound")
    angular_lower = 1 - 2 * t - 4 * (144 + 14) * sigma_radius
    require(angular_lower > Fraction(9, 10)
            and 4 * (144 + 14) / angular_lower < 1000, "Angular equation")
    require(8 * 1000 * sigma_radius < Fraction(1, 4), "Full-turn continuation")
    require(t**8 < sigma_radius / 2, "Taylor remainder disk")
    error = (2 * 10**7 + 1 / sigma_radius**2) * t
    # Use pi > 3 and 1/sqrt(6) > 2/5.
    leading_lower = 3 * Fraction(2, 5) * disk_moment(3) * Fraction(3, 4)**3 / 4
    require(error < Fraction(1, 200) < leading_lower, "Sample scalar bound is below the model root-disk margin")


def run_checks():
    for d in range(1, 11):
        for n in (10, 100):
            check_moments(d, Fraction(1, n))
            for s in range(1, 7):
                expanded_field(s, d, n)
    check_specified_field(10**20)
    check_error_bounds()
    print("Exact integrand, disk moments, collected supports and sample-scale scalar inequalities passed.")
    print("At t=10^-20 the eight-term field check is algebraic. U2 proves the cycle count for sufficiently small t.")
    small = {4: 12, 5: 12, 6: 12, 7: 16, 8: 20, 9: 24, 10: 32}
    print("m   GS Theorem 2   GS Proposition 1   U2 small-scale bound")
    for m in range(4, 101):
        proposition, family = benchmark(m)
        require(family == benchmark(m + 1)[0], "Adjacent-budget identity")
        if m >= 8:
            require(family > max(proposition, small.get(m, 0)), "Fixed-budget improvement")
        if m <= 16:
            print(f"{m:2}  {str(small.get(m, '-')):>12}  {str(proposition or '-'):>17}  {family:>12}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--field", nargs=3, type=int, metavar=("S", "D", "N"),
                        help="print the expanded field for s centers, d cycles and N")
    args = parser.parse_args()
    if args.field is None:
        run_checks()
        return
    try:
        p, q = expanded_field(*args.field)
    except ValueError as error:
        parser.error(str(error))
    for name, component in (("P", p), ("Q", q)):
        print(f"{name} has {len(component)} terms (x exponent, y exponent, coefficient)")
        for (x_power, y_power), coefficient in sorted(component.items()):
            print(x_power, y_power, coefficient)


if __name__ == "__main__":
    main()
