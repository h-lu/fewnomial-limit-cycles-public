"""Exact rational checks and expanded component supports for flat Hamiltonian minima.

Run python3 -B -m research.flat_minima.verify from the repository root.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import cache
from math import comb, factorial, lcm, prod

from research.exponent_variations.verify import binomial_derivative, inverse_and_determinant, matvec
from research.saddle_moments.verify import print_field
from research.separated_exponents.verify import family_bound, interval_add_scaled
from research.verify_term_bounds import lift_and_collect, require


@cache
def binomial(x, order):
    return prod(x - j for j in range(order)) * Q(1, factorial(order))


@cache
def endpoint_entry(n, order, j):
    y = Q(2 * j + 1, 2 * n)
    return Q(2, 2 * j + 1) * binomial(y, order) / (y - order)


def endpoint_index(n, index):
    group, position = divmod(index, n + 1)
    if position == 0:
        return group, False, n * group, 2 * n * group, (-1)**group
    lowest = n * group + position - 1
    return group, True, lowest, 2 * lowest + 1, (-1)**(group + 1)


@cache
def basis(n, index):
    group, fractional, lowest, _, _ = endpoint_index(n, index)
    columns = range(lowest + fractional, index + 1)
    matrix = [[endpoint_entry(n, order, j) for j in columns] for order in range(group + 1)]
    rhs = [(-endpoint_entry(n, order, lowest) if fractional else Q(order == group))
           for order in range(group + 1)]
    inverse, _ = inverse_and_determinant(matrix)
    return tuple([Q(0)] * lowest + [Q(1)] * fractional + matvec(inverse, rhs))


def density(n, d, q):
    limit = endpoint_index(n, d)[3] + 3
    values = [Q(0)] * (d + 1)
    for index in range(d + 1):
        _, _, _, exponent, sign = endpoint_index(n, index)
        weight = (-1)**index * sign * q**((limit - exponent)**2)
        for j, coefficient in enumerate(basis(n, index)):
            values[j] += weight * coefficient
    return values


def model_dyadic_exponent(n, d):
    norm = max(Q(1), *(sum(map(abs, basis(n, i))) for i in range(d + 1)))
    bound = 8 * (d + 1) * (2 * d + 1) * (d + 2) * 2**(d + 6) * n * norm
    # Smallest integer s with 2**s > bound.
    return (bound.numerator // bound.denominator).bit_length()


def lower_interval(n, root, coefficients):
    """Here h=root**(2n). Integrate 1 <= 1/(1-s) <= 1/(1-h)."""
    h = root**(2 * n)
    value = Q(0), Q(0)
    for j, c in enumerate(coefficients):
        y = Q(2 * j + 1, 2 * n)
        term = 2 * h * root**(2 * j + 1) / ((2 * j + 1) * (1 + y))
        value = interval_add_scaled(value, (term, term / (1 - h)), c)
    return value


def upper_interval(n, root, coefficients):
    """Regular series from vI'-yI=-(1-v)**y, plus the exact fractional powers."""
    v = root**(2 * n)
    last = len(coefficients) + 2
    value = Q(0), Q(0)
    for j, c in enumerate(coefficients):
        y = Q(2 * j + 1, 2 * n)
        theta = y - int(y)
        regular = sum((-v)**ell * endpoint_entry(n, ell, j) for ell in range(last + 1))
        # Beyond this index both |binom(y,ell)| and 1/|y-ell| decrease.
        tail = abs(endpoint_entry(n, last + 1, j)) * v**(last + 1) / (1 - v)
        value = interval_add_scaled(value, (regular - tail, regular + tail), c)
        # Split integral x**(theta-1)/(1+x) at x=1.
        ceiling = 1 / theta + 1 / (1 - theta)
        factor = Q(2 * (-1)**(int(y) + 1), 2 * j + 1) * root**(2 * j + 1)
        value = interval_add_scaled(value, (ceiling / 2, ceiling), c * factor)
    return value


def changes(n, d):
    raw = 2 * d - d // (n + 1) - 1
    return raw + raw % 2


def modular_determinant(rows, prime):
    rows = [[entry % prime for entry in row] for row in rows]
    result = 1
    for col in range(len(rows)):
        pivot = next((i for i in range(col, len(rows)) if rows[i][col]), None)
        if pivot is None:
            return 0
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            result = -result
        value = rows[col][col]
        result = result * value % prime
        reciprocal = pow(value, -1, prime)
        for i in range(col + 1, len(rows)):
            if rows[i][col]:
                factor = rows[i][col] * reciprocal % prime
                for j in range(col + 1, len(rows)):
                    rows[i][j] = (rows[i][j] - factor * rows[col][j]) % prime
                rows[i][col] = 0
    return result


def check_budget_rank(n, p):
    """Obtain the unscaled Taylor matrix a second way, via series inversion."""
    prime = 65537
    d = 3 * p - 2 * n
    inverse_int = [0] + [pow(i, -1, prime) for i in range(1, 2 * d + 2)]
    even = list(range(0, 2 * p, 2))
    top = 2 * p - 1
    extra = list(range(2 * n, 2 * p, 2)) + [top]
    # Subtract L^j from the highest active even degree down until the base is odd.
    B = [comb(top, i) % prime for i in range(top + 1)]
    for j in reversed(even):
        coefficient = B[j]
        for ell in range(j + 1):
            B[ell] = (B[ell] - coefficient * comb(j, ell)) % prime
    require(not any(B[::2]) and B[-1] == 1, "Odd base in second calculation")
    rows = []
    for i in range(d + 1):
        power = 2 * i
        row = [comb(s, power) for s in even]
        for s in extra[:-1]:
            # [Y^power]D(z(Y))z'(Y) = [z^power]D(z)(2nV/z^(2n))^(-(power+1)/(2n)).
            value = 0
            for ell in range(2 * n - 1, s + 1):
                other = power + 2 * n - 1 - ell
                if 0 <= other <= top:
                    value += B[other] * comb(s, ell) * inverse_int[ell + 1]
            row.append((power + 1) * value)
        for s in extra:
            row.append(sum(comb(s, j) * (-1)**(power - j + 1) * inverse_int[power - j]
                           for j in range(min(s + 1, power))))
        rows.append(row)
    det = modular_determinant(rows, prime)
    first = 2 * p - n
    fact = [factorial(j) % prime for j in range(2 * d + 1)]
    factorial_minor = [[fact[2 * i - 1 - s] for s in extra] for i in range(first, d + 1)]
    product_det = modular_determinant(factorial_minor, prime)
    for s in extra[:-1]:
        product_det *= (2 * p + s - 2 * n + 1) * inverse_int[s + 1]
    for s in extra:
        product_det *= (-1)**(s + 1) * fact[s]
    product_det *= pow(prod(fact[2 * i] for i in range(first, d + 1)), -1, prime)
    require(det == product_det % prime and det != 0, "Full finite-budget rank, two derivations")
    print(f"n={n}, p={p}: {d+1} by {d+1} determinant = {det} modulo {prime}")


def check_sign_marks(n, d, q):
    coefficients = density(n, d, q)
    limit = endpoint_index(n, d)[3] + 3
    energies, signs = [], []
    for index in range(d + 1):
        _, fractional, lowest, exponent, _ = endpoint_index(n, index)
        if fractional:
            root = q**(2 * (limit - exponent))
            sign = (-1)**lowest
            bounds = lower_interval(n, root, coefficients)
            require(min(sign * x for x in bounds) > 0, "Lower sign at fixed rational mark")
            energies.append(root**(2 * n))
            signs.append(sign)
    for index in reversed(range(d + 1)):
        exponent = endpoint_index(n, index)[3]
        root = q**(2 * (limit - exponent))
        sign = (-1)**index
        bounds = upper_interval(n, root, coefficients)
        require(min(sign * x for x in bounds) > 0, "Upper sign at fixed rational mark")
        energies.append(1 - root**(2 * n))
        signs.append(sign)
    require(all(a < b for a, b in zip(energies, energies[1:])), "Disjoint ordered marks")
    require(sum(a != b for a, b in zip(signs, signs[1:])) == changes(n, d), "Sign count")
    print(f"n={n}, d={d}, q={q}: {len(energies)} rational marks, {changes(n,d)} sign changes")


@dataclass
class Shape:
    even: list
    extra: list
    weights: dict
    matrix: list
    inverse: list


@cache
def shape(n, p):
    if n < 2 or p <= n:
        raise ValueError("Require n >= 2 and p >= n+1.")
    r0, d = 2 * n - 1, 3 * p - 2 * n
    even = list(range(0, 2 * p, 2))
    top = 2 * p - 1
    extra = list(range(2 * n, 2 * p, 2)) + [top]
    weights = {top: 1}
    for a in reversed(even):
        weights[a] = -sum(c * comb(j, a) for j, c in weights.items())
    B = [sum(c * comb(j, degree) for j, c in weights.items()) for degree in range(2 * p)]
    require(not any(B[::2]) and B[-1] == 1 and all(weights.values()), "Odd monic base density")
    # Coefficients of -z^(-r0) times the primitive of L^s minus its Taylor polynomial.
    h = {s: {ell + 1 - r0: -Q(comb(s, ell), ell + 1) for ell in range(r0, s + 1)}
         for s in extra}
    matrix = []
    for i in range(d + 1):
        degree = 2 * i
        transported = {s: -(degree + 1) * sum(c * h[s].get(degree + 1 - j, 0)
                                              for j, c in enumerate(B)) for s in extra}
        require(sum(weights[s] * transported[s] for s in extra) == 0, "Even offset cancels")
        row = [Q(comb(j, degree)) for j in even] + [transported[s] for s in extra[:-1]]
        row += [binomial_derivative(s, degree) for s in extra]
        matrix.append(row)
    inverse, determinant = inverse_and_determinant(matrix)
    require(determinant != 0 and len(inverse) == d + 1, "Full vertical control rank")
    return Shape(even, extra, weights, matrix, inverse)


def expanded_field(n, p, k, q, delta, shift, t, epsilon):
    if k < 1 or not all(0 < val < 1 for val in (q, delta, t)) or epsilon <= 0:
        raise ValueError("Require positive k and epsilon and 0 < q,delta,t < 1.")
    system = shape(n, p)
    d, r0 = 3 * p - 2 * n, 2 * n - 1
    target = density(n, d, q)
    target[0] += shift
    rhs = [c * t**(2 * (d - i)) for i, c in enumerate(target)]
    controls = matvec(system.inverse, rhs)
    require(matvec(system.matrix, controls) == rhs, "Exact vertical jets")
    a = dict(zip(system.even, controls[:p]))
    sigma = dict(zip(system.extra[:-1], controls[p:2 * p - n]))
    eta = dict(zip(system.extra, controls[2 * p - n:]))
    norm = max(sum(map(abs, row)) for row in system.inverse)
    offset = int(2 + norm * max(map(abs, target))) + 1
    lam = t**(2 * d + 2)
    powers = {j: Q(j) for j in range(r0 + 1)}
    amplitudes = {j: Q((-1)**(r0 - j + 1) * comb(r0, j)) for j in range(r0 + 1)}
    for s in system.extra:
        powers[s] = s + lam * eta[s] / system.weights[s]
        amplitudes[s] = lam * (sigma.get(s, 0) + offset * system.weights[s])
        for ell in range(r0):
            correction = amplitudes[s] * binomial(powers[s], ell)
            for j in range(ell + 1):
                amplitudes[j] -= correction * (-1)**(ell - j) * comb(ell, j)
    require(len(set(powers.values())) == len(powers) and all(amplitudes.values()), "Distinct active powers")
    # Independent Taylor expansion, after collecting in L, at z=0.
    jets = [sum(amplitudes[j] * binomial(power, ell) for j, power in powers.items())
            for ell in range(r0 + 1)]
    require(not any(jets[:-1]) and jets[-1] < 0, "Exact flat minimum and its sign")
    alpha = {j: epsilon * (system.weights.get(j, 0) + lam * a.get(j, 0)) / amplitudes[j]
             for j in powers}
    rates = [delta**i for i in range(1, k + 1)]
    scale = 2 * lcm(*(val.denominator for val in [*powers.values(), *alpha.values(), *rates]))
    first = [(amplitudes[j], alpha[j], 1 + power) for j, power in powers.items()]
    g = t**(2 * n) / (2 * n)
    second = [(g * sum(rates), 1, 0)] + [(-g * b, b, 0) for b in rates]
    result = lift_and_collect(first, scale, 1, 0), lift_and_collect(second, scale, 0, 1)
    require(tuple(map(len, result)) == (p + n + 1, k + 1), "Exact reduced component supports")
    return result


def check_ranks():
    for n in range(2, 7):
        for index in range(19):
            a, fractional, lowest, _, _ = endpoint_index(n, index)
            coefficients = basis(n, index)
            require(not any(coefficients[:lowest]) and coefficients[-1], "Endpoint basis degree")
            if fractional:
                require(coefficients[lowest] == 1, "Fractional leading coefficient")
            for ell in range(a + 1):
                value = sum(c * endpoint_entry(n, ell, j) for j, c in enumerate(coefficients))
                require(value == (ell == a and not fractional), "Endpoint cancellations")
        for p in range(n + 1, n + 5):
            system = shape(n, p)
            # Independently factor the full determinant using the triangular polynomial block.
            i0 = 2 * p - n
            leading = prod(Q(2 * p + s - 2 * n + 1, s + 1) for s in system.extra[:-1])
            size = len(system.extra)
            factorial_rows = [[Q(factorial(2 * (i0 + i) - 1 - s)) for s in system.extra]
                              for i in range(size)]
            tail = inverse_and_determinant(factorial_rows)[1]
            tail *= prod((-1)**(s + 1) * factorial(s) for s in system.extra)
            tail /= prod(factorial(2 * (i0 + i)) for i in range(size))
            require(inverse_and_determinant(system.matrix)[1] == leading * tail, "Factorial minor")
    print("Endpoint basis through degree 18 at n=2..6; twenty vertical matrices and factorial minors.")


def check_budgets():
    for m, n, p, k, expected, old in [(50, 2, 24, 22, 10032, 9800), (100, 3, 48, 47, 45496, 42112)]:
        d = 3 * p - 2 * n
        require(m == p + n + k + 2 and 4 * k * changes(n, d) == expected, "Finite allocation")
        require(family_bound(m) == old, "Audited envelope")
        check_budget_rank(n, p)
        print(f"m={m}: ({p+n+1},{k+1}) terms, {4*k*changes(n,d)} cycles, previous {old}")
    for n, p, k in [(2, 3, 1), (2, 4, 3), (3, 5, 2), (4, 5, 3)]:
        field = expanded_field(n, p, k, Q(1, 16), Q(1, 64), Q(1, 2**40), Q(1, 64), Q(1, 64**160))
        require(tuple(map(len, field)) == (p + n + 1, k + 1), "Separate support reproduction")
    print("Four rational field expansions and their exact minimum orders checked (support examples).")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--field", nargs=8, metavar=("N", "P", "K", "Q", "DELTA", "SHIFT", "T", "EPS"),
                       help="emit exact exponent pairs and rational coefficients at the supplied parameters")
    group.add_argument("--model", nargs=2, metavar=("N", "D"), type=int,
                       help="compute the sufficient dyadic exponent for q from the model basis norms")
    args = parser.parse_args()
    if args.field:
        n, p, k, q, delta, shift, t, epsilon = args.field
        print_field(expanded_field(int(n), int(p), int(k), *map(Q, (q, delta, shift, t, epsilon))))
    elif args.model:
        n, d = args.model
        if n < 2 or d < 1:
            parser.error("Require n >= 2 and d >= 1.")
        print(f"n={n}, d={d}, sufficient model q = 2^-{model_dyadic_exponent(n, d)}")
    else:
        check_ranks()
        for n, d in [(2, 8), (3, 9), (4, 10)]:
            check_sign_marks(n, d, Q(1, 256))
        check_budgets()


if __name__ == "__main__":
    main()
