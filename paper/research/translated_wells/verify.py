"""Exact arithmetic for the translated-well construction. Run as a module."""

from argparse import ArgumentParser
from fractions import Fraction as Q
from math import isqrt

from research.chebyshev_rows.verify import chebyshev
from research.filtered_outlets.verify import at, comparators, evaluate, sine_key
from research.flat_minima.verify import changes, check_sign_marks, modular_determinant
from research.mixed_outlet.verify import polynomial_product
from research.translated_wells.construction import (
    affine, controls, cycle_count, expanded_field, fit_primitive, layers, model_target,
)
from research.verify_term_bounds import require


def check_confluent_minor(p, b):
    """Derivative of the binomial recurrence versus the factorial alternant."""
    prime = 65537
    K, D = 2*p-b, 3*p-b-1
    fact = [1]
    inv = [0] + [pow(i, -1, prime) for i in range(1, D+1)]
    for i in range(1, D+1):
        fact.append(i*fact[-1] % prime)
    columns = []
    for j in range(2, p+1):
        coefficient, derivative = 1, 0
        column = []
        for h in range(1, D+1):
            derivative = ((j+1-h)*derivative+coefficient)*inv[h] % prime
            coefficient = (j+1-h)*coefficient*inv[h] % prime
            if h > K:
                column.append(derivative)
        columns.append(column)
    det = modular_determinant(list(zip(*columns)), prime)
    q = p-1
    expected = (-1)**(q*(q-1)//2)
    for i, h in enumerate(range(K+1, D+1)):
        expected = expected*(-1)**(h-1)*fact[h-p-1]*pow(fact[h], -1, prime)*fact[i] % prime
    for j in range(2, p+1):
        expected = expected*(-1)**j*fact[j] % prime
    require(det == expected != 0, "Confluent minor and reversed monic-polynomial alternant")
    return det


def check_fit():
    n, L, a = 2, 2, 1
    p, b, eta = 2*n*L, 2*n*a, Q(1, 2**128)
    D = 3*p-b-1
    target = chebyshev(D)
    target[2] += 7
    polynomial, logs = fit_primitive(target, p, b, eta)
    shape = controls(n, L, a, eta, polynomial, logs)
    # Independent Taylor series through D, using differentiation of X^j log X
    # in the exponent by its binomial recurrence.
    got = affine(polynomial, 1, eta) + [Q(0)]*(p-1)
    for j in range(2, p+1):
        coefficient, derivative = Q(1), Q(0)
        for h in range(1, D+1):
            derivative = ((j+1-h)*derivative+coefficient)/h
            coefficient = (j+1-h)*coefficient/h
            got[h] += logs[j]*derivative*eta**h
    require(got == target, "Independent ordinary Taylor readback through the last degree")
    # On |z| <= 1, coefficients beyond D are bounded by sum 2^j |logs_j|
    # in w. Differentiating twice costs at most h^2. For eta <= 1/2 the
    # sum of (l+1)^2 eta^l is at most 12.
    tail = 12*(D+1)**2 * sum(2**j*abs(v) for j, v in enumerate(logs))*eta**(D+1)
    require(tail < Q(1, 2**32), "Uniform rational C^2 tail bound")
    require(any(shape.exponent_change[2:b+1]) and any(shape.exponent_change[b+1:]),
            "The fit uses exponents of nonzero and of absent baseline force terms")
    print(f"Degree {D} reproduced, with rational C^2 error below 2^-32 on [-1,1].")


def check_moving_well():
    n, L, a, eta, t = 4, 2, 1, Q(1, 16), Q(1, 16)
    p, b = 2*n*L, 2*n*a
    # This example combines actual shifted endpoint polynomials and both row parities.
    T = model_target(n, L, a, [Q(1, 4), Q(1, 8)], [Q(1, 18), Q(-1, 19)], t, Q(1, 8))
    poly, logs = fit_primitive(T, p, b, eta)
    sh = controls(n, L, a, eta, poly, logs)
    level_values = []
    for z in [Q(3, 5), Q(4, 5)]:
        X = 1+eta*z
        v0, f0 = evaluate(sh.potential, X), evaluate(sh.primitive, X)
        c = 1-evaluate(chebyshev(n), z)**2
        require(v0 == c**a and f0 == c**L, "Read both baseline primitives in physical z")
        level_values.append((v0, f0))
        vz, fz = (eta*sum(j*v*X**(j-1) for j, v in enumerate(part) if j)
                  for part in (sh.potential, sh.primitive))
        # Evaluate constant and log(X) parts separately at the moving slice endpoints.
        for perturb, base, expected in (
            ((sh.potential_change, sh.primitive_change), False, poly),
            ((sh.potential, sh.primitive), True, logs),
        ):
            dV, dF = (sum(cj*X**j*(sh.exponent_change[j] if base else 1)
                          for j, cj in enumerate(part)) for part in perturb)
            velocity = -dV/vz
            require(velocity != 0 and dF+fz*velocity == evaluate(expected, X),
                    "Branch motion and static variation reproduce the effective primitive")
    require(level_values[0] == level_values[1], "Two branches of the same well at equal energy")
    field = expanded_field(sh, a, 3, Q(1, 16), t)
    require(tuple(map(len, field)) == (4, 16), "Two-block expansion from the model polynomials")
    print("Two rational moving branches and the full two-block support (4,16) passed.")


def filtered_bound(m):
    """Enumerate both grids and every integer split in pinned Theorem 1."""
    best = 0
    for grid in ("integer", "half"):
        n = 4
        while n <= m-2:
            for L in range(1, (m-2)//n+1):
                k = m-1-n*L
                for a in range(1, L+1):
                    S = n*(2*L-a)
                    g, total = n, 0
                    while g >= 4:
                        if grid == "integer":
                            rows, next_g = g//2, g//2
                        elif g > 4:
                            rows, next_g = 2*g//3, g//3
                        else:
                            rows, next_g = 4, 1
                        total += rows*changes(a, 2*S//g-1)
                        g = next_g
                    best = max(best, 2*k*total)
            n *= 2 if grid == "integer" else 3
    return best


def best_translated(m):
    values = []
    n = 2
    while 4*n <= m-2:
        for L in range(2, (m-2)//(2*n)+1):
            k = m-1-2*n*L
            values.extend(cycle_count(n, L, a, k) for a in range(1, L))
        n *= 2
    return max(values, default=0)


def saddle_formula_ceiling(m):
    """Credit the optional outer crossing at every starting degree in the pinned formula."""
    best = 0
    for n in range(1, (m-4)//4+1):
        d = 3*(m-2-4*n)
        per_pair = max(4*((3*s+1)//4) + s//2+1 + d-s for s in range(1, d+1))
        best = max(best, 4*n*per_pair)
    return best


def support_example(n, L, a, k):
    """Full-size coefficient and support check with small prescribed shape changes."""
    p = 2*n*L
    eta = Q(1, 16)
    zero = controls(n, L, a, eta, [0], [Q(0)]*(p+1))
    requested = [Q(0), Q(0)] + [Q((-1)**j, j+1) for j in range(2, p+1)]
    weighted_v = [v*requested[j] for j, v in enumerate(zero.potential)]
    product = polynomial_product(weighted_v, zero.quotient)
    logs = [v*requested[j]-at(product, j) for j, v in enumerate(zero.primitive)]
    K = 2*p-2*n*a
    poly = [Q((-1)**j*(j+1)) for j in range(K+1)]
    sh = controls(n, L, a, eta, poly, logs)
    require(sh.exponent_change == requested, "Triangular solve recovers all supplied slopes")
    field = expanded_field(sh, a, k, Q(1, 4), Q(1, 8))
    require(tuple(map(len, field)) == (k+1, p), "Collected table-budget field")


def check_budgets():
    cases = [
        (45, 8, 2, 1, 12, 8160, 6720, 7904),
        (50, 8, 2, 1, 17, 11560, 8400, 10032),
        (120, 16, 2, 1, 55, 100760, 67680, 67200),
        (180, 16, 3, 1, 83, 245016, 173808, 158688),
        (324, 32, 3, 1, 131, 973592, 638352, 542880),
    ]
    for m, n, L, a, k, total, previous, flat in cases:
        p, S = 2*n*L, n*(3*L-a)
        stages = layers(n, S)
        rows_seen, modes_seen = set(), set()
        per_shelf = 0
        for stage in stages:
            rows, modes = set(stage.rows), set(stage.frequencies)
            require(rows_seen.isdisjoint(rows) and modes_seen.isdisjoint(modes)
                    and len(rows)*len(modes) == S and max(modes) <= 2*S-1, "Block counts")
            for i in rows:
                require(all(sine_key(h*i, n) == 0 for h in modes_seen), "Prior frequencies vanish")
                weights = [sine_key(h*i, n)*s for h, s in zip(stage.frequencies, stage.signs)]
                require(len(set(weights)) == 1 and weights[0] != 0, "Signed sine weights factor")
            rows_seen.update(rows)
            modes_seen.update(modes)
            per_shelf += len(rows)*changes(a, len(modes)-1)
        require(rows_seen == set(range(1, n)) and len(stages) == n.bit_length()-1,
                "Every interior well, including the central one, is used")
        require(p+k+1 == m and 4*k*per_shelf == total == best_translated(m),
                "Exact budget, counted returns and integer optimization")
        require(filtered_bound(m) == previous, "Optimized pinned filtered outlets")
        old_flat, _, separated, gs = comparators(m)
        saddle = saddle_formula_ceiling(m)
        require(old_flat == flat and total > max(previous, flat, separated, gs, saddle),
                "Equal-budget comparisons")
        minor = check_confluent_minor(p, 2*n*a)
        support_example(n, L, a, k)
        print(f"m={m}, support ({k+1},{p}), {per_shelf} per shelf per quadrant, "
              f"{total:,} cycles; filtered {previous:,}, flat {flat:,}, separated {separated:,}, "
              f"saddle formula at most {saddle:,}, GS Prop. 1 {gs:,}; log minor {minor} mod 65537.")


def check_allocations():
    for m in range(4096, 24001):
        u = (m-1).bit_length()
        n = 1 << ((m//u**2).bit_length()-1)
        L = (m-1)//(4*n)
        p, a = 2*n*L, isqrt(L)
        k = m-1-p
        require(1 <= a < L and n*u**2 <= m < 2*n*u**2 and k >= 1, "All-budget parameters")
        require((m-1)**2-4*k*p == (k-p)**2 < (4*n)**2, "Exact product and rounding error")
    print("Every exact allocation from 4,096 through 24,000 passed.")


def check_cross_minima():
    """Equality examples for the supplied univariate support bound on the pinned cross."""
    for p in range(1, 7):
        cheb = chebyshev(2*p)
        center = cheb[0]
        for a, b, interior in [(0, 0, ()), (2, 1, (Q(1, 3),)), (1, 2, (Q(-1, 4),)*2)]:
            derivative = [j*cheb[j] for j in range(1, len(cheb))]
            for level in (center,)*a + (-center,)*b + interior:
                factor = [cheb[0]-level] + cheb[1:]
                derivative = polynomial_product(derivative, factor)
            count = p+(p-1)*a+p*(b+len(interior))
            occupied = [j for j, value in enumerate(derivative) if value]
            require(occupied == list(range(2*a+1, 2*a+1+2*count, 2)),
                    "Multiplicity, omitted low powers and exact collected support")
    print("Univariate equality examples for the supplied invisible-cross bounds passed.")


def field_from_args(args):
    n, L, a, k = map(int, args[:4])
    cycle_count(n, L, a, k)
    delta, t, zeta, eta = map(Q, args[6:10])

    def per_block(spec):
        values = list(map(Q, spec.split(",")))
        return values*(n.bit_length()-1) if len(values) == 1 else values

    T = model_target(n, L, a, per_block(args[4]), per_block(args[5]), t, zeta)
    poly, logs = fit_primitive(T, 2*n*L, 2*n*a, eta)
    sh = controls(n, L, a, eta, poly, logs)
    lam, mu = (None if value == "auto" else Q(value) for value in args[10:])
    return expanded_field(sh, a, k, delta, t, lam, mu)


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model-50", action="store_true",
                       help="Enclose the signs of all three 50-term endpoint models.")
    group.add_argument("--field", nargs=12, metavar=(
        "N", "L", "A", "K", "Q", "SHIFT", "DELTA", "T", "ZETA", "ETA", "LAMBDA", "MU"),
        help="Print all collected hexadecimal exponent and coefficient rows for supplied parameters.")
    args = parser.parse_args()
    if args.model_50:
        for stage in layers(8, 40):
            check_sign_marks(1, len(stage.frequencies)-1, Q(1, 256))
        return
    if args.field:
        for name, component in zip(("P", "Q"), field_from_args(args.field)):
            print(name)
            for (x, y), c in sorted(component.items()):
                print(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}")
        return
    check_fit()
    check_moving_well()
    check_budgets()
    check_allocations()
    check_cross_minima()


if __name__ == "__main__":
    main()
