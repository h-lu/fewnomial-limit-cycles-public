"""Exact arithmetic for Chebyshev rows with separated energy intervals.

Run python3 -B -m research.filtered_outlets.verify from the repository root.
The sample rational fields test expansion. The cycle theorem selects parameters
by existence.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import cache
from math import isqrt, lcm

from research.chebyshev_rows.verify import chebyshev, cycle_count, primitive
from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.flat_minima.verify import (
    changes, check_sign_marks, density, model_dyadic_exponent, modular_determinant,
)
from research.mixed_outlet.verify import polynomial_product
from research.separated_exponents.verify import family_bound
from research.verify_term_bounds import benchmark, require


def powers(poly, exponent):
    result = [1]
    for _ in range(exponent):
        result = polynomial_product(result, poly)
    return result


def trim(poly):
    while len(poly) > 1 and not poly[-1]:
        poly.pop()
    return poly


def even_well(grid, n):
    """Coefficients in z=x^2 of the actual polynomial C(x)."""
    square = polynomial_product(chebyshev(n), chebyshev(n))
    require(not any(square[1::2]), "Even Chebyshev well")
    c = square[::2]
    if grid == "integer":
        c = [-value for value in c]
        c[0] += 1
    elif grid != "half":
        raise ValueError("Choose half or integer.")
    return c


def divide_primitive(f, divisor):
    """Return R,F0 in F=-C^L R+F0, with ordinary polynomial division in z."""
    remainder = trim(list(map(Q, f)))
    quotient = [Q(0)] * max(1, len(remainder) - len(divisor) + 1)
    for offset in range(len(remainder) - len(divisor), -1, -1):
        factor = remainder[offset + len(divisor) - 1] / divisor[-1]
        quotient[offset] = -factor
        for index, value in enumerate(divisor):
            remainder[offset + index] -= factor * value
    return trim(quotient), trim(remainder)


def at(poly, j):
    return poly[j] if 0 <= j < len(poly) else 0


def evaluate(poly, z):
    value = 0
    for coefficient in reversed(poly):
        value = value * z + coefficient
    return value


def first_derivative(poly, x):
    """x derivative of a primitive supplied in z=x^2, at the rational point x."""
    return 2*x*evaluate([j*poly[j] for j in range(1, len(poly))], x*x)


def forces(c, L, a, f, t, lam):
    n = len(c) - 1
    p = n * L
    if not 1 <= a <= L or len(f) > n*(2*L-a)+1 or t <= 0:
        raise ValueError("Inadmissible potential order, primitive degree or spatial scale.")
    ca, cl = powers(c, a), powers(c, L)
    r, remainder = divide_primitive(f, cl)
    v1 = polynomial_product(ca, r)
    require(len(r) <= n*(L-a)+1 and len(remainder) <= p and len(v1) <= p+1,
            "Actual degrees after division and multiplication")
    g = [-t**(-2*a) * (2*j*(at(ca, j) + lam*at(v1, j)) + lam**2)
         for j in range(1, p+1)]
    h = [Q(2*a*j, L)*at(cl, j) + 2*j*lam*at(remainder, j) for j in range(1, p+1)]
    return g, h


def collect(terms):
    result = {}
    for x, y, coefficient in terms:
        result[x, y] = result.get((x, y), 0) + coefficient
    return {key: value for key, value in result.items() if value}


def expanded_field(g, h, k, delta, mu):
    if not (0 < delta < 1 and k >= 1 and mu > 0 and len(g) == len(h) and all(g)):
        raise ValueError("Require positive regular parameters and nonzero restoring coefficients.")
    rates = [delta**j for j in range(1, k+1)]
    beta = [mu*v/w for w, v in zip(g, h)]
    if max(map(abs, beta)) >= Q(1, 2):
        raise ValueError("Exponent tilt outside the sufficient range.")
    D = 2*lcm(*(b.denominator for b in rates+beta))

    def scaled(exponent):
        value = D * exponent
        require(value.denominator == 1 and value >= 0 and value % 2 == 0, "Even integral lift")
        return int(value)

    p = collect([(0, 2*D, D*sum(rates))] + [(0, scaled(1+b), -D*b) for b in rates])
    q = collect([(2*j+1, 1+scaled(1+b), coeff) for j, (b, coeff) in enumerate(zip(beta, g))])
    require((len(p), len(q)) == (k+1, len(g)), "Independently reduced real component supports")
    # Read the logarithmic rates and their first variation back from the real exponent lists.
    require({(Q(y, D)-1, v/D) for (_, y), v in p.items()}
            == {(Q(1), sum(rates))} | {(b, -b) for b in rates}, "Horizontal pullback readback")
    for (power, y), coeff in q.items():
        i = (power-1)//2
        require(coeff == g[i] and coeff*(Q(y-1, D)-1) == mu*h[i], "Vertical pullback readback")
    return p, q


@dataclass(frozen=True)
class Block:
    modulus: int
    rows: tuple
    modes: tuple
    signs: tuple


def sine_key(angle, n):
    """Encode sin(pi*angle/n) by sign and the folded numerator, without rounding."""
    phase = angle % (2*n)
    if phase in (0, n):
        return 0
    sign = 1 if phase < n else -1
    angle = phase if phase < n else phase - n
    return sign * min(angle, n-angle)


def blocks(grid, n, S):
    if S < n or S % n:
        raise ValueError("The stated family requires a positive multiple of n modes.")
    if grid == "half":
        survivors = list(range(1, 2*n, 2))
    elif grid == "integer":
        survivors = list(range(2, 2*n, 2))
    else:
        raise ValueError("Choose half or integer.")
    g = n
    result, earlier = [], set()
    while g >= 4:
        if grid == "half" and g > 4:
            if g % 3:
                raise ValueError("Half-shifted family requires n=4*3^r.")
            residue, next_g, count = g//3, g//3, 2*g//3
        else:
            if g % 4:
                raise ValueError("Integer-phase family requires n=2^r.")
            residue = g//4
            next_g, count = (0, 4) if grid == "half" else (g//2, g//2)
        modes = tuple(s for s in range(1, S+1) if s % g in (residue, g-residue))
        exposed = tuple(b for b in survivors if sine_key(modes[0]*b, n))
        exposed_set = set(exposed)
        blind = [b for b in survivors if b not in exposed_set]
        first = exposed[0]
        signs = tuple(1 if sine_key(s*first, n) > 0 else -1 for s in modes)
        require(len(modes) == 2*S//g and len(exposed) == count and earlier.isdisjoint(modes),
                "Independent stage counts and disjoint frequencies")
        for b in exposed:
            require(all(s*b % n == 0 for s in earlier), "Earlier stages vanish on this row")
            scale = signs[0]*sine_key(modes[0]*b, n)
            require(all(sine_key(s*b, n) == scale*e for s, e in zip(modes, signs)),
                    "Exact signed column replication")
        require(all(s*b % n == 0 for b in blind for s in modes), "Exact remaining blind set")
        result.append(Block(g, exposed, modes, signs))
        earlier.update(modes)
        survivors, g = blind, next_g
    require(survivors == ([] if grid == "half" else [n]), "Terminal rows")
    return result


def jet(frequency, n, size):
    alpha = Q(2*frequency, n)
    values = [Q(1)]
    for j in range(1, size):
        values.append(values[-1]*((2*j-1)**2-alpha**2)/((2*j)*(2*j-1)))
    return values


def target_primitive(n, layers, polynomials, t, zeta):
    """Exact primitive F for supplied shifted model polynomials and rational scales."""
    size = max(s for block in layers for s in block.modes)
    f = [Q(0)] * (size+1)
    for ell, block in enumerate(layers, 1):
        M = len(block.modes)
        a = polynomials[ell-1]
        if len(a) > M:
            raise ValueError("Model polynomial exceeds its block degree.")
        matrix = list(map(list, zip(*(jet(s, n, M) for s in block.modes))))
        rhs = [at(a, j)/t**(2*j) for j in range(M)]
        v = matvec(inverse_and_determinant(matrix)[0], rhs)
        require(matvec(matrix, v) == rhs, "Exact interpolation of every assigned coefficient")
        for s, coeff, sign in zip(block.modes, v, block.signs):
            weight = zeta**ell * sign * coeff
            for j in range(1, s+1):
                f[j] += weight * primitive(s, j)
    return trim(f)


def check_rank(n, block):
    """Construct f via cos(alpha*arcsin u) times the inverse square root.

    Compare the determinant of that convolution to the leading-coefficient
    Vandermonde product, in a prime field. A nonzero residue proves full rank.
    """
    prime = 65537
    M = len(block.modes)
    inv = [0] + [pow(j, -1, prime) for j in range(1, 2*M+1)]
    root = [1]
    leading = [1]
    for j in range(1, M):
        root.append(root[-1]*(2*j-1)*inv[2*j] % prime)
        leading.append(-leading[-1]*inv[2*j]*inv[2*j-1] % prime)
    columns, nodes = [], []
    for s in block.modes:
        node = (2*s*pow(n, -1, prime))**2 % prime
        nodes.append(node)
        cosine = [1]
        for j in range(M-1):
            cosine.append(cosine[-1]*((2*j)**2-node)*inv[2*j+1]*inv[2*j+2] % prime)
        columns.append([sum(cosine[i]*root[j-i] for i in range(j+1)) % prime for j in range(M)])
    det = modular_determinant(list(zip(*columns)), prime)
    expected = 1
    for j, node in enumerate(nodes):
        expected = expected * leading[j] % prime
        for other in nodes[:j]:
            expected = expected * (node-other) % prime
    require(det == expected != 0, "Taylor determinant versus independent cosine convolution")
    return det


def compact_mu(g, h):
    return Q(1, 4)/(1+max(abs(v/w) for w, v in zip(g, h)))


def sample_primitive(c, L, a):
    """A prescribed top-degree shape direction and a dense remainder for expansion checks."""
    n, p = len(c)-1, (len(c)-1)*L
    quotient = [1] + [0]*(n*(L-a))
    quotient[-1] += 2
    f = polynomial_product(powers(c, L), quotient)
    f[0] = Q(0)
    for j in range(1, p):
        f[j] += Q((-1)**j, 2*j)
    return f


def check_moving_slice():
    # In C=1-T4^2, 3/5 and 4/5 lie on opposite sides of cos(pi/4).
    n, L, a = 4, 3, 2
    c = even_well("integer", n)
    cl, ca = powers(c, L), powers(c, a)
    f = sample_primitive(c, L, a)
    r, rem = divide_primitive(f, cl)
    v1 = polynomial_product(ca, r)
    left, right = Q(3, 5), Q(4, 5)
    require(evaluate(c, left**2) == evaluate(c, right**2) > 0 and
            first_derivative(c, left)*first_derivative(c, right) < 0, "Two inverse branches")

    def moving_endpoint(x):
        d0 = Q(a, L)*first_derivative(cl, x)
        velocity = -evaluate(v1, x*x)/first_derivative(ca, x)
        return d0*velocity

    motion = moving_endpoint(right) - moving_endpoint(left)
    static = evaluate(rem, right**2) - evaluate(rem, left**2)
    require(motion and static and motion+static == evaluate(f, right**2)-evaluate(f, left**2),
            "Exact moving-boundary slice derivative including nonzero endpoint motion")
    print("Two rational inverse branches give the prescribed moving-slice derivative exactly.")


def check_mode_field():
    grid, n, L, a, k = "integer", 8, 2, 1, 2
    layers = blocks(grid, n, n*(2*L-a))
    model = [density(a, len(block.modes)-1, Q(1, 256)) for block in layers]
    for j, poly in enumerate(model):
        poly[0] += Q(1, 2**(4096+j))
    f = target_primitive(n, layers, model, Q(1, 16), Q(1, 2**16))
    c = even_well(grid, n)
    require(len(divide_primitive(f, powers(c, L))[0]) > 1, "Mode target uses moving ovals")
    g, h = forces(c, L, a, f, Q(1, 16), Q(1, 17))
    return expanded_field(g, h, k, Q(1, 8), compact_mu(g, h))


def comparators(m):
    flat = max((4*(m-p-a-2)*changes(a, 3*p-2*a) for a in range(2, m//2+1)
                for p in range(a+1, m-a-2)), default=0)
    grid = 0
    n, r = 3, 1
    while 2*n <= m:
        grid = max(grid, cycle_count(n, r, m+1-2*n))
        n, r = 3*n, r+1
    return flat, grid, family_bound(m), benchmark(m)[0]


def check_budgets():
    cases = [(120, 36, 2, 1, 47, 67680, 67200, 5494, 6960),
             (180, 36, 3, 1, 71, 173808, 158688, 10414, 15840),
             (324, 36, 5, 1, 143, 638352, 542880, 57376, 51840)]
    for m, n, L, a, k, total, old_flat, old_rows, gs in cases:
        layers = blocks("half", n, n*(2*L-a))
        per_level = sum(len(b.rows)*changes(a, len(b.modes)-1) for b in layers)
        require(2*k*per_level == total and n*L+k+1 == m, "Finite theorem allocation")
        dets = [check_rank(n, block) for block in layers]
        c = even_well("half", n)
        g, h = forces(c, L, a, sample_primitive(c, L, a), Q(1, 8), Q(1, 17))
        field = expanded_field(g, h, k, Q(1, 8), compact_mu(g, h))
        require(tuple(map(len, field)) == (k+1, n*L), "Full budget expanded with exact rationals")
        flat, grid, separated, published = comparators(m)
        require((flat, grid, published) == (old_flat, old_rows, gs)
                and total > max(flat, grid, separated, published), "Equal-budget comparators")
        print(f"m={m}: ({k+1},{n*L}) terms, blocks (rows,modes) "
              f"{[(len(b.rows), len(b.modes)) for b in layers]}, det residues {dets}")
        print(f"  {total:,} cycles; flat {flat:,}, earlier rows {grid:,}, "
              f"separated {separated:,}, GS Prop. 1 {published:,}")
    require(cases[-1][5] - 6*324**2 == 8496, "Strict excess over 6m^2")


def check_uniform_allocation():
    for m in range(4096, 24001):
        u = (m-1).bit_length()
        n = 1 << ((m//u**2).bit_length()-1)
        L = (m-1)//(2*n)
        a = isqrt(L)
        p, k = n*L, m-1-n*L
        require(4 <= n and n*u**2 <= m < 2*n*u**2 and 1 <= a <= L and k >= 1,
                "Integer dyadic choice at every exact budget")
        require((m-1)**2 - 4*k*p == (k-p)**2 < 4*n**2, "Rounded budget product")
    for r in range(2, 9):
        n = 2**r
        for L, a in [(1, 1), (3, 2)]:
            S = n*(2*L-a)
            stages = blocks("integer", n, S)
            require(sum(len(b.rows)*len(b.modes) for b in stages) == S*(r-1),
                    "Dyadic work including the missing central row")
    sample = blocks("integer", 16, 64)
    require(all(check_rank(16, block) for block in sample), "Integer-phase Taylor ranks")
    print("Dyadic visibility through n=256 and every integer allocation at m=4096..24000 passed.")


def max_filter_work(grid, n, S):
    """Optimize the specified signed-column recipe on actual sets of live row labels."""
    labels = tuple(range(1, 2*n, 2) if grid == "half" else range(0, 2*n, 2))

    @cache
    def best(survivors):
        # Select all modes with the same folded column up to a common sign.
        # On the residual cyclic grids nonzero proportional columns differ only by this sign.
        classes = {}
        for s in range(1, S+1):
            column = tuple(sine_key(s*b, n) for b in survivors)
            first = next((w for w in column if w), 0)
            if not first:
                continue
            key = column if first > 0 else tuple(-w for w in column)
            classes[key] = classes.get(key, 0) + 1
        value = 0
        for key, count in classes.items():
            blind = tuple(b for b, w in zip(survivors, key) if not w)
            value = max(value, count*(len(survivors)-len(blind)) + best(blind))
        return value

    return best(labels)


def check_work_limit():
    for grid, radix, exponents in [("half", 3, range(1, 5)), ("integer", 2, range(2, 7))]:
        for r in exponents:
            n = radix**r
            for S in [n, 2*n]:
                maximum = max_filter_work(grid, n, S)
                expected = 4*S*r//3 if grid == "half" else S*(r-1)
                require(maximum == expected, "Enumerated optimal work on the actual sine columns")
    print("Exact optimal mode copies by exhaustive stage choice through n=81 (half) and 64 (integer).")


def field_from_arguments(values):
    grid = values[0]
    n, L, a, k = map(int, values[1:5])
    model_scales, shifts = values[5:7]
    delta, t, zeta, lam, mu = map(Q, values[7:])
    if not (n >= 4 and 1 <= a <= L and k >= 1 and 0 < t < 1 and 0 < zeta < 1 and lam > 0):
        raise ValueError("Inadmissible construction parameters.")
    layers = blocks(grid, n, n*(2*L-a))

    def layer_values(spec):
        vals = list(map(Q, spec.split(",")))
        if len(vals) == 1:
            return vals*len(layers)
        if len(vals) != len(layers):
            raise ValueError("Use one rational value or one value per block.")
        return vals

    polynomials = []
    for block, q, shift in zip(layers, layer_values(model_scales), layer_values(shifts)):
        if not 0 < q < 1:
            raise ValueError("Model q must lie between zero and one.")
        polynomial = density(a, len(block.modes)-1, q)
        polynomial[0] += shift
        polynomials.append(polynomial)
    f = target_primitive(n, layers, polynomials, t, zeta)
    g, h = forces(even_well(grid, n), L, a, f, t, lam)
    return expanded_field(g, h, k, delta, mu)


def main():
    parser = ArgumentParser(description=__doc__)
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--example-field", action="store_true", help="print a support sample, in hexadecimal")
    tasks.add_argument("--model-120", action="store_true", help="reproduce sufficient model scales at m=120")
    tasks.add_argument("--field", nargs=12,
                       metavar=("GRID", "N", "L", "A", "K", "Q", "SHIFT", "DELTA", "T", "ZETA", "LAM", "MU"),
                       help="collect the field for supplied parameters; Q and SHIFT allow comma-separated lists")
    args = parser.parse_args()
    if args.model_120:
        for d, expected in [(5, 29), (17, 53), (53, 114)]:
            exponent = model_dyadic_exponent(1, d)
            require(exponent == expected, "Rational endpoint-basis norm at this degree")
            print(f"d={d}, sufficient scale for the limiting density q=2^-{exponent}")
        return
    if args.field or args.example_field:
        field = field_from_arguments(args.field) if args.field else check_mode_field()
        for name, component in zip(("P", "Q"), field):
            print(name, "(x exponent, y exponent, coefficient numerator, denominator), base 16")
            for (x, y), value in sorted(component.items()):
                print(*(format(i, "x") for i in (x, y, value.numerator, value.denominator)))
        return
    for a, d in [(1, 5), (2, 7)]:
        check_sign_marks(a, d, Q(1, 256))
    check_moving_slice()
    require(tuple(map(len, check_mode_field())) == (3, 16), "Collected two-block support example")
    check_budgets()
    check_uniform_allocation()
    check_work_limit()


if __name__ == "__main__":
    main()
