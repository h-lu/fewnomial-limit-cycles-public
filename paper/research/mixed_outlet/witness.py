"""Rational enclosures and finite-scale inequalities for the specified 100-cycle field."""

from fractions import Fraction as Q
from functools import cache
from math import comb, factorial
from pathlib import Path

from research.mixed_outlet.verify import DELTA, EPSILON_POWER, T, coefficients, controls
from research.separated_exponents.verify import interval_add_scaled, sqrt_bounds
from research.verify_term_bounds import require

BITS = 1024
GRID = 2**BITS
RADIUS = Q(1, 2**512)


def positive_series(x, last, ratio, tail):
    """Round the successive positive terms down and up on a common dyadic grid."""
    lo = hi = total_lo = total_hi = GRID
    for i in range(last):
        multiplier = x * ratio(i)
        a, b = multiplier.numerator, multiplier.denominator
        lo = lo * a // b
        hi = -((-hi * a) // b)
        total_lo += lo
        total_hi += hi
    return Q(total_lo, GRID), Q(total_hi, GRID) + tail


@cache
def half_pi():
    def atan(n):
        lo = sum(Q((-1) ** i, (2 * i + 1) * n ** (2 * i + 1)) for i in range(240))
        return lo, lo + Q(1, 481 * n**481)

    return interval_add_scaled(interval_add_scaled((0, 0), atan(5), 8), atan(239), -2)


def exponential_interval(h, c):
    value = Q(0), Q(0)
    if h <= Q(1, 2):
        root = sqrt_bounds(h, BITS)
        for j, a in enumerate(c):
            bounds = positive_series(
                h, 1024, lambda i, degree=j: Q(2 * i + 2, 2 * i + 2 * degree + 5), Q(1, GRID)
            )
            weight = 4 * a * h ** (j + 1) / ((2 * j + 1) * (2 * j + 3))
            value = interval_add_scaled(value, (root[0] * bounds[0], root[1] * bounds[1]), weight)
        return value

    v = 1 - h
    require(0 < v <= Q(1, 16), "Upper brackets in range of the arcsine series")
    regular = singular = 0
    for j, a in enumerate(c):
        scale = 2 * a / (2 * j + 1)
        regular += scale * sum((-v) ** i * h ** (j - i) * Q(2, 2 * (j - i) + 1) for i in range(j + 1))
        singular += 2 * scale * (-1) ** (j + 1) * v**j
    sine = positive_series(v, 256, lambda i: Q((2 * i + 1) ** 2, (2 * i + 2) * (2 * i + 3)), Q(1, GRID))
    pi, root = half_pi(), sqrt_bounds(v, BITS)
    trig = pi[0] * root[0] - v * sine[1], pi[1] * root[1] - v * sine[0]
    return interval_add_scaled(
        interval_add_scaled(value, sqrt_bounds(h, BITS), regular), trig, singular
    )


def linear_interval(h, c):
    value = sum(4 * a * h ** (j + 1) / ((2 * j + 1) * (2 * j + 3)) for j, a in enumerate(c))
    return interval_add_scaled((0, 0), sqrt_bounds(h, BITS), value)


def check_models():
    c = coefficients()
    regular, singular = [Q(0)] * 8, [Q(0)] * 8
    for j, a in enumerate(c):
        scale = 2 * a / (2 * j + 1)
        singular[j] = 2 * scale * (-1) ** (j + 1)
        for i in range(j + 1):
            for k in range(j - i + 1):
                regular[i + k] += scale * (-1) ** (i + k) * comb(j - i, k) * Q(2, 2 * (j - i) + 1)
    for polynomial in (regular, singular):
        for order in range(3):
            norm = sum(
                abs(polynomial[j]) * Q(factorial(j), factorial(j - order)) for j in range(order, 8)
            )
            require(norm < 2**25, "Coefficient norm of analytic prefactor and its derivatives")
    counts = {}
    name = None
    for line in Path(__file__).with_name("brackets.txt").read_text().splitlines():
        if line in ("exp", "linear"):
            name, previous = line, Q(0)
            counts[name] = 0
            continue
        center = Q(int(line, 16), 2**576)
        lo, hi = center - RADIUS, center + RADIUS
        cutoff = 1 - Q(1, 2**112) if name == "exp" else 16
        require(max(previous, Q(1, 2**90)) < lo < hi < cutoff, "Regular disjoint brackets")
        for mark, sign in [(lo, (-1) ** counts[name]), (hi, (-1) ** (counts[name] + 1))]:
            bounds = (exponential_interval if name == "exp" else linear_interval)(mark, c)
            require(min(sign * x for x in bounds) > Q(1, 2**710), "Opposite endpoint signs with margin")
        previous = hi
        counts[name] += 1
    require(counts == {"exp": 10, "linear": 5}, "Exactly the claimed model brackets were examined")
    require(Q(1, 2**710) / RADIUS - 2 * RADIUS * 2**220 > Q(1, 2**200), "Slope throughout each bracket")
    print(
        "15 disjoint model brackets in their respective ranges; |moment| > 2^-710 at the endpoints, |slope| > 2^-200 throughout."
    )


def check_transfer():
    q, root_delta, tau = DELTA, Q(1, 2**4096), Q(1, 2**2048)
    require(root_delta**2 == q and tau**4 == q, "Chosen cutoffs")
    eta = 512 * q
    require(eta + Q(1, 2**128) < Q(1, 2**112) and 2 * root_delta < Q(1, 2**91), "Turning points")
    require(2**64 * root_delta < Q(1, 2**4000), "Finite horizontal value error")
    require(
        2**105 * root_delta + 2**246 * Q(1, 2**1024) + 128 * 2**50 * eta * 2**3072 < Q(1, 2**770),
        "Finite horizontal derivative error",
    )
    kappa, _, cn, rn = controls()
    require(abs(kappa) < 1 and cn < 2**60 and rn < 2**60, "Bounds for the vertical parameters")
    require(2**230 * T * T * 2**200 < Q(1, 2**15000), "Finite vertical errors")
    # Compare binary exponents, avoiding gigantic integers in the ODE estimates.
    require(
        EPSILON_POWER == 1024 and 8 + 8192 * (400 + 29 - EPSILON_POWER) < -1000,
        "Exact one-turn C1 error",
    )
    print(
        "Horizontal, vertical and full return estimates preserve all twenty-five brackets in one quadrant."
    )


def main():
    check_models()
    check_transfer()


if __name__ == "__main__":
    main()
