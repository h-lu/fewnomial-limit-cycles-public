"""Rational enclosures for one eight-term field with 96 hyperbolic limit cycles."""

from argparse import ArgumentParser
from fractions import Fraction as R
from math import comb, factorial
from pathlib import Path
import gzip
import hashlib
import sys

from research.separated_exponents.verify import (
    density, expanded_field, interval_add_scaled, print_field, sqrt_bounds,
)
from research.verify_term_bounds import require


DELTA = R(1, 2**4096)
T = DELTA
EPSILON = T**100
GRID = 2**640
RADIUS = R(1, 2**330)
NUMERATORS = (
    667495932273871344798647361974527077561863009944984500395404398077406431779002998059709699192698,
    1337041813043312953040858568835591694475715810548510566879322353811668635074708174295416230585976073725757,
    996042617238334110014622705519278667876046294958581677207017143227956751650167088539909006177948157835751103713233,
    2462625296221572798806234633582616370045831891775396536554794046917337517980113745430353411482356515172881851707931,
    2462625387264640813512915880749090288109604725425711326510588855801708868513439536897648693357130188051337244644791,
    2462625387274654943787829644000779319532844487506931976947870460973105862080564295504738785930005688310163961217366,
    2462625387274654950744422561837220391470180831929041177179047872620621394208863000804228174026673049458904387777239,
    2462625387274654950767440006057378450478831950121121374232921335326519955540120197651670994367284061531289062276144,
)


def atan_inverse(n):
    total = sum(R((-1)**i, (2 * i + 1) * n**(2 * i + 1)) for i in range(160))
    return total, total + R(1, 321 * n**321)


def half_pi():
    return interval_add_scaled(interval_add_scaled((0, 0), atan_inverse(5), 8),
                               atan_inverse(239), -2)


def positive_series(x, last, ratio, tail):
    """Enclose a positive series, rounding each term outwards on the dyadic grid."""
    low = high = sum_low = sum_high = GRID
    for i in range(last):
        multiplier = x * ratio(i)
        a, b = multiplier.numerator, multiplier.denominator
        low = low * a // b
        high = -((-high * a) // b)
        sum_low += low
        sum_high += high
    return R(sum_low, GRID), R(sum_high, GRID) + tail


def value_interval(h, coefficients):
    if h <= R(1, 2):
        root = sqrt_bounds(h, 640)
        total = (R(0), R(0))
        for j, c in enumerate(coefficients):
            series = positive_series(h, 640, lambda i: R(2 * i + 2, 2 * i + 2 * j + 5),
                                     R(1, 2**640))
            factor = 4 * c * h**(j + 1) / ((2 * j + 1) * (2 * j + 3))
            total = interval_add_scaled(total, (root[0] * series[0], root[1] * series[1]), factor)
        return total

    v = 1 - h
    require(0 < v < R(1, 2**20), "All upper root enclosures use small v")
    regular = singular = 0
    for j, c in enumerate(coefficients):
        scale = 2 * c / (2 * j + 1)
        regular += scale * sum((-v)**l * h**(j - l) * R(2, 2 * (j - l) + 1)
                               for l in range(j + 1))
        singular += 2 * scale * (-1)**(j + 1) * v**j
    # sqrt(v) arccos(sqrt(v)) = sqrt(v) pi/2 - v sum binom(2i,i)v^i/(4^i(2i+1)).
    sine_series = positive_series(v, 32, lambda i: R((2 * i + 1)**2, (2 * i + 2) * (2 * i + 3)),
                                  R(1, 2**640))
    pi = half_pi()
    root = sqrt_bounds(v, 640)
    trig = pi[0] * root[0] - v * sine_series[1], pi[1] * root[1] - v * sine_series[0]
    return interval_add_scaled(interval_add_scaled((0, 0), sqrt_bounds(h, 640), regular),
                               trig, singular)


def check_curvature(coefficients):
    # Coefficients of P(v), Q(v) in J = sqrt(1-v) P(v) + sqrt(v) arccos(sqrt(v)) Q(v).
    regular = [R(0)] * 6
    singular = [R(0)] * 6
    for j, c in enumerate(coefficients):
        scale = 2 * c / (2 * j + 1)
        singular[j] = 2 * scale * (-1)**(j + 1)
        for i in range(j + 1):
            for k in range(j - i + 1):
                regular[i + k] += scale * (-1)**(i + k) * comb(j - i, k) * R(2, 2 * (j - i) + 1)
    for poly in (regular, singular):
        for order in range(3):
            require(sum(abs(poly[j]) * R(factorial(j), factorial(j - order))
                        for j in range(order, 6)) < 2**20, "Polynomial derivative norm")
    require(max(map(abs, coefficients)) < 32, "Coefficient bound for h <= 1/2")


def check_roots(coefficients):
    last = 0
    for index, numerator in enumerate(NUMERATORS):
        center = R(numerator, 2**380)
        lo, hi = center - RADIUS, center + RADIUS
        require(max(last, R(1, 2**81)) < lo < hi < 1 - R(1, 2**97), "Regular disjoint root brackets")
        for h, direction in [(lo, (-1)**index), (hi, (-1)**(index + 1))]:
            bounds = value_interval(h, coefficients)
            require(min(direction * x for x in bounds) > R(1, 2**465), "Strict oriented endpoint margin")
        last = hi
    require(R(1, 2**465) / RADIUS - 2 * RADIUS * 2**180 > R(1, 2**136),
            "Derivative lower bound from slopes and curvature")


def check_transfers():
    tau = R(1, 2**1024)
    scale_error = 390 * DELTA
    require(scale_error + R(1, 2**128) < R(1, 2**97), "Turning point at x < 128")
    require(scale_error < tau and 2**30 * R(1, 2**2048) < R(1, 2**2000), "Area error")
    derivative_error = (2**105 * R(1, 2**2048) + 2**212 * R(1, 2**512)
                        + 128 * 2**20 * scale_error * 2**1536)
    require(derivative_error < R(1, 2**298), "First derivative error at finite exponents")
    require(2**320 * T**2 < R(1, 2**2000), "Transported vertical error")
    ru, local, w_radius = (R(1, 2**i) for i in (29000, 84000, 97000))
    require(32 * ru < R(1, 2**28701), "Complex square-root branch")
    require(2**70400 * local < R(1, 2**12502)
            and w_radius < local / 2**12501, "Uniform horizontal inverse disk")
    energy_radius = R(1, 2**100000)
    epsilon_radius = R(1, 2**117000)
    require(4 * energy_radius * 2**42 < w_radius, "Complex energy disk")
    require(8 * 2**16420 * epsilon_radius < energy_radius, "One-turn radial change")
    require(2 * EPSILON < epsilon_radius and
            2 * EPSILON / (T**21 * epsilon_radius**2) < R(1, 2**1000),
            "Full return and derivative Taylor errors")


def field():
    return expanded_field(3, 2, DELTA, T, EPSILON, density(5, R(1, 16)), offset=10**6)


def fingerprint(components):
    """Hash a canonical sign/length/magnitude binary encoding for very large integers."""
    digest = hashlib.sha256()
    for first, part in zip((True, False), components):
        digest.update(b'P' if first else b'Q')
        for (x, y), c in sorted(part.items()):
            for integer in (x, y, c.numerator, c.denominator):
                magnitude = abs(integer).to_bytes(max(1, (abs(integer).bit_length() + 7) // 8), 'big')
                digest.update(bytes((int(integer < 0),)) + len(magnitude).to_bytes(8, 'big') + magnitude)
    return digest.hexdigest()


def check_export(components):
    raw = gzip.decompress(Path(__file__).with_name("field.txt.gz").read_bytes())
    require(hashlib.sha256(raw).hexdigest() ==
            "fc95797ed7545472d05d29b485336251a9918411b8cf1a2b6fc9d4f3becbe5b9",
            "Exact archive text digest")
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    lines = raw.decode("ascii").splitlines()
    require(len(lines) == 10 and lines[0] == "P" and lines[5] == "Q", "Archive component format")
    for component, rows in zip(components, (lines[1:5], lines[6:])):
        triples = []
        for row in rows:
            i, j, coefficient = row.split()
            triples.append(((int(i), int(j)), R(coefficient)))
        require(triples == sorted(component.items()), "All expanded exponents and coefficients")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--field", action="store_true", help="print all eight exact decimal triples")
    args = parser.parse_args()
    if args.field:
        print_field(field())
        return
    coefficients = density(5, R(1, 16))
    check_curvature(coefficients)
    check_roots(coefficients)
    check_transfers()
    components = field()
    check_export(components)
    print("Eight disjoint rational brackets have both endpoint margins > 2^-465 and |J'| > 2^-136.")
    print("Finite horizontal, vertical, inverse-coordinate and full-return inequalities passed.")
    print("Archived exponent and coefficient triples match. Fixed support", tuple(map(len, components)))
    print("Binary field fingerprint", fingerprint(components))


if __name__ == "__main__":
    main()
