"""Rational bracket and finite-return checks for the 108-cycle witness."""

from fractions import Fraction as Q
from math import comb, factorial
from pathlib import Path

from research.mixed_outlet.verify import DELTA, EPSILON_POWER, T
from research.mixed_outlet.witness import RADIUS, exponential_interval, linear_interval
from research.tail_interpolation.verify import Q_MODEL, controls, target
from research.verify_term_bounds import require


def check_curvature(c):
    regular, singular = [Q(0)] * 8, [Q(0)] * 8
    for j, a in enumerate(c):
        scale = 2*a/(2*j+1)
        singular[j] = 2*scale*(-1)**(j+1)
        for i in range(j+1):
            for k in range(j-i+1):
                regular[i+k] += scale * (-1)**(i+k) * comb(j-i, k) * Q(2, 2*(j-i)+1)
    for poly in (regular, singular):
        for order in range(3):
            norm = sum(abs(a)*Q(factorial(j), factorial(j-order))
                       for j, a in enumerate(poly) if j >= order)
            require(norm < 2**25, "Analytic prefactor derivative bound")


def check_brackets():
    c, _, _ = target()
    q = Q_MODEL
    common = (q**18, q**10, q**2, Q(1, 20))
    marks = {
        "exp": (*common, Q(1, 2), *(1-Q(1, 2**j) for j in (10, 30, 45, 62, 75, 100))),
        "linear": (*common, Q(3, 4), Q(19, 20), Q(1), Q(5, 4)),
    }
    methods = {"exp": exponential_interval, "linear": linear_interval}
    for name, points in marks.items():
        for i, h in enumerate(points):
            require(min((-1)**i*z for z in methods[name](h, c)) > 0, "Alternating coarse marks")
    check_curvature(c)

    name = None
    counts = {}
    min_margin = Q(1)
    for line in Path(__file__).with_name("brackets.txt").read_text().splitlines():
        if line in methods:
            require(line not in counts, "Unique bracket header")
            name = line
            counts[name] = 0
            continue
        i = counts[name]
        center = Q(int(line, 16), 2**576)
        lo, hi = center-RADIUS, center+RADIUS
        require(marks[name][i] < lo < hi < marks[name][i+1], "One bracket per sign pair")
        require(Q(1, 2**90) < lo and hi < (1-Q(1, 2**112) if name == "exp" else 16),
                "Compact ranges for horizontal transfer")
        for h, sign in ((lo, (-1)**i), (hi, (-1)**(i+1))):
            margin = min(sign*z for z in methods[name](h, c))
            min_margin = min(min_margin, margin)
        counts[name] += 1
    require(counts == {"exp": 10, "linear": 7}, "Model bracket counts")
    require(min_margin > Q(1, 2**710), "Uniform endpoint margin")
    require(Q(1, 2**710)/RADIUS - 2*RADIUS*2**220 > Q(1, 2**200),
            "Uniform derivative from secants and curvature")
    print("10 exponential and 7 linear brackets, endpoint margin > 2^-710 and |slope| > 2^-200.")


def check_transfer():
    c, _, _ = target()
    kappa, _, control_norm, residual_norm = controls()
    require(abs(kappa) < 1 and max(residual_norm, control_norm) < 2**60,
            "Bounds on these particular vertical controls")
    require(sum(abs(v)*32**j for j, v in enumerate(c)) < 2**48
            and sum(j*abs(v)*32**(j-1) for j, v in enumerate(c) if j) < 2**48,
            "Density bounds on all vertical slices")
    root_delta, cutoff = Q(1, 2**4096), Q(1, 2**2048)
    require(root_delta**2 == cutoff**4 == DELTA == T, "Fixed rates and cutoffs")
    eta = 512*DELTA
    require(eta + Q(1, 2**128) < Q(1, 2**112) and 2*root_delta < Q(1, 2**91),
            "Turning-point range at all marks of the narrow brackets")
    require(2**64*root_delta < Q(1, 2**4000), "Horizontal value error")
    require(2**105*root_delta + 2**246*Q(1, 2**1024) + 128*2**50*eta*2**3072
            < Q(1, 2**770), "Horizontal derivative error")
    require(2**230*T*T*2**200 < Q(1, 2**15000), "Vertical transport error")
    require(EPSILON_POWER == 1024 and 8+8192*(429-EPSILON_POWER) < -1000,
            "Normalized nonlinear return error")
    print("At the specified field parameters the signs and slopes survive on 10 + 10 + 7 brackets.")


if __name__ == "__main__":
    check_brackets()
    check_transfer()
