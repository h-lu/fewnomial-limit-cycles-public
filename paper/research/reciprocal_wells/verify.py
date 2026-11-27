"""Independent arithmetic checks. Run from the repository root as a module."""

from argparse import ArgumentParser
from fractions import Fraction as Q

from research.chebyshev_rows.verify import chebyshev
from research.filtered_outlets.verify import at, comparators, evaluate, sine_key
from research.mixed_outlet.verify import polynomial_product
from research.reciprocal_wells.construction import (
    baseline, best_parameters, controls, cycle_count, field_from_models, high_model, layers,
    model_count, model_target,
)
from research.reciprocal_wells.endpoint_check import certified_model
from research.translated_wells.construction import affine, expanded_field, fit_primitive
from research.translated_wells.verify import (
    best_translated, check_confluent_minor, filtered_bound, saddle_formula_ceiling,
)
from research.verify_term_bounds import require


def modular_rank(rows, prime):
    rows = [list(row) for row in rows]
    rank = 0
    for j in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][j]), None)
        if pivot is None:
            continue
        rows[pivot], rows[rank] = rows[rank], rows[pivot]
        inv = pow(rows[rank][j], -1, prime)
        for i in range(rank+1, len(rows)):
            factor = rows[i][j]*inv % prime
            if factor:
                for k in range(j, len(rows[i])):
                    rows[i][k] = (rows[i][k]-factor*rows[rank][k]) % prime
        rank += 1
    return rank


def check_visible_rank(n, L):
    """Odd Taylor rows at phi=pi/n in the rational local parameter u=tan((theta-phi)/2)."""
    prime, p = 65537, n*L
    K, expected = 2*p-n, 3*p-n-2*L
    order = 2*(expected+p)+1
    inv = lambda a: pow(a, -1, prime)
    omega = pow(3, (prime-1)//(2*n), prime)
    imaginary = pow(3, (prime-1)//4, prime)
    require(pow(omega, n, prime) == prime-1 and imaginary**2 % prime == prime-1,
            "Exact order and square root in the finite field")
    cosine = (omega+inv(omega))*inv(2) % prime
    sine = (omega-inv(omega))*inv(2*imaginary) % prime
    # X = 1 + eta*((1-u^2)*cosine - 2*u*sine)/(1+u^2).
    eta = inv(16)
    X = [0]*(order+1)
    for j, value in enumerate((1+eta*cosine, -2*eta*sine, 1-eta*cosine)):
        X[j] = value
    for j in range(order+1):
        X[j] = (X[j]-(X[j-2] if j >= 2 else 0)) % prime
    require(X[0] != 0, "Translation unit")

    def multiply(left, right):
        return [sum(left[i]*right[j-i] for i in range(j+1)) % prime for j in range(order+1)]

    # Coefficients of log(X/X(0)) from X*(log X)'=X'. The omitted constant
    # changes a logarithmic column by a polynomial column.
    log = [0]*(order+1)
    for j in range(1, order+1):
        lower = sum(i*log[i]*X[j-i] for i in range(1, j))
        log[j] = (j*X[j]-lower)*inv(j*X[0]) % prime
    power = [1]+[0]*order
    columns = [power[1::2]]
    for j in range(1, K+1):
        power = multiply(power, X)
        columns.append(power[1::2])
        if 2 <= j <= p:
            columns.append(multiply(power, log)[1::2])
    rank = modular_rank(zip(*columns), prime)
    require(rank == expected and len(columns)-rank == 2*L, "One-well visible and blind dimensions")
    print(f"n={n}, L={L}: reflected odd-jet rank {rank} of {len(columns)} modulo {prime}.")


def model_sample():
    n, L, a, k = 4, 2, 1, 3
    kappa, t, eta = Q(1, 4), Q(1, 8), Q(1, 16)
    models = []
    for ell, block in enumerate(layers(n, L)):
        c = high_model(a, len(block.frequencies), Q(1, 4))
        c[0] += Q((-1)**ell, 17)*(a+kappa)
        models.append(c)
    return n, L, a, k, models, kappa, Q(1, 16), t, Q(1, 8), eta


def check_moving_branches():
    n, L, a, k, models, kappa, delta, t, zeta, eta = model_sample()
    p = n*L
    target = model_target(n, L, a, t, zeta, models)
    poly, logs = fit_primitive(target, p, n, eta)
    shape = controls(n, L, a, t, kappa, eta, poly, logs)
    levels, differences = [], []
    for z in (Q(3, 5), Q(4, 5)):
        X = 1+eta*z
        v0, f0, D = (evaluate(part, X) for part in (shape.potential, shape.primitive, shape.denominator))
        J = evaluate(chebyshev(n), z)
        require(f0 == J and v0 == (1-J**2)*(1+t**2+kappa), "Physical baseline on each branch")
        levels.append((v0, D))
        vprime, fprime = (sum(j*c*X**(j-1) for j, c in enumerate(part) if j)
                          for part in (shape.potential, shape.primitive))
        require(vprime == D*fprime != 0, "Reciprocal derivative on two moving slice endpoints")
        effects = []
        for potentials, exponent_part, goal in (
            ((shape.potential_change, shape.primitive_change), False, poly),
            ((shape.potential, shape.primitive), True, logs),
        ):
            dV, dF = (sum(c*X**j*(shape.exponent_change[j] if exponent_part else 1)
                           for j, c in enumerate(part)) for part in potentials)
            effect = dF-fprime*dV/vprime
            require(effect*D == evaluate(goal, X), "Transported numerator on the moving endpoint")
            effects.append(effect)
        differences.append(effects)
    require(levels[0] == levels[1] and differences[0] != differences[1], "Active paired branches")
    field = expanded_field(shape, a, k, delta, t)
    require(tuple(map(len, field)) == (4, 8), "Expanded two-block sample")
    # Independent Taylor readback using the exponent derivative of binomial(j,h).
    fit_eta = Q(1, 2**128)
    degree = 3*p-n-1
    goal = chebyshev(degree)
    poly, logs = fit_primitive(goal, p, n, fit_eta)
    recovered = affine(poly, 1, fit_eta) + [Q(0)]*(p-1)
    for j in range(2, p+1):
        coeff, tangent = Q(1), Q(0)
        for h in range(1, degree+1):
            tangent = ((j-h+1)*tangent+coeff)/h
            coeff = (j-h+1)*coeff/h
            recovered[h] += logs[j]*tangent*fit_eta**h
    require(recovered == goal, "All translated Taylor rows")
    tail = 12*(degree+1)**2*sum(2**j*abs(c) for j, c in enumerate(logs))*fit_eta**(degree+1)
    require(tail < Q(1, 2**32), "Rational uniform C^2 numerator error on [-1,1]")
    print(f"Two active rational branches, separate support (4,8), degree-{degree} C^2 fit passed.")


def check_second_order():
    """Read a curved invisible path and its order-two transverse perturbation on two branches."""
    eta, t, kappa = Q(1, 16), Q(1, 8), Q(1, 4)
    inner = affine(chebyshev(4), -1/eta, 1/eta)

    def derivative(poly):
        return [j*poly[j] for j in range(1, len(poly))]

    def node_jets(data, X, ell):
        # Coefficients of order one and two at fixed X for shared nodes j*(1+lambda).
        first, second, c1, dc1, c2 = data
        return (c1+X*ell*first, dc1+(1+ell)*first+X*ell*second,
                c2+X*ell*dc1+ell**2*(X*first+X**2*second)/2)

    branches = []
    for X in (1+eta*Q(3, 5), 1+eta*Q(4, 5)):
        I, Ix, Ixx = (evaluate(part, X) for part in (inner, derivative(inner), derivative(derivative(inner))))
        # I_lambda=I+lambda*U, A_lambda=A+lambda*(1+s), B_lambda=s+lambda*(1+s^2).
        # A(s)=(1-s^2)*(1+kappa+t^2*s) has odd degree three.
        U, Ux = 1+X**2, 2*X
        Ap = -2*(1+kappa)*I+t**2*(1-3*I**2)
        App = -2*(1+kappa)-6*t**2*I
        V = (Ap*Ix, App*Ix**2+Ap*Ixx, Ap*U+1+I, App*Ix*U+Ap*Ux+Ix, App*U**2/2+U)
        F = (Ix, Ixx, U+1+I**2, Ux+2*I*Ix, 2*I*U)
        require(X*Ix+X**2*Ixx != 0, "A nonzero raw log-squared primitive coefficient")
        results = []
        # The order-two expressions have degree at most two in the formal log(X).
        for ell in (0, 1, 2):
            v1, v1x, v2 = node_jets(V, X, ell)
            f1, f1x, f2 = node_jets(F, X, ell)
            dx = -v1/V[0]
            d2x = -(v2+v1x*dx+V[1]*dx**2/2)/V[0]
            first = f1+F[0]*dx
            second = f2+f1x*dx+F[1]*dx**2/2+F[0]*d2x
            results.append((first, second, second+X))
        require(len(set(results)) == 1, "Logarithmic terms cancel after moving the endpoint")
        branches.append(results[0])
    left, right = branches
    require(left[:2] == right[:2] and right[2]-left[2] == Q(1, 80),
            "A transverse lambda^2 X term recovers precisely its first-variation difference")
    print("Quadratic node jets cancel on a curved blind path. A transverse order-two gap is 1/80.")


def check_table_support(n, L, a, k):
    p, eta, t, kappa = n*L, Q(1, 16), Q(1, 8), Q(1, 4)
    v, f, D = [affine(part, -1/eta, 1/eta) for part in baseline(n, L, a, t, kappa)]
    require(len(v)-1 == p and len(D)-1 == p-n, "Exact translated degrees")
    expected = [Q(0), Q(0)] + [Q((-1)**j, j+1) for j in range(2, p+1)]
    product = polynomial_product([cj*expected[j] for j, cj in enumerate(f)], D)
    logs = [at(product, j)-v[j]*expected[j] for j in range(p+1)]
    poly = [Q((-1)**j*(j+1)) for j in range(2*p-n+1)]
    sh = controls(n, L, a, t, kappa, eta, poly, logs)
    require(sh.exponent_change == expected, "Triangular reconstruction of all test slopes")
    polynomial = polynomial_product(sh.denominator, sh.primitive_change)
    require([at(polynomial, j)-at(sh.potential_change, j) for j in range(len(poly))] == poly,
            "Polynomial numerator reconstructed after ordinary collection")
    P, Qpart = expanded_field(sh, a, k, Q(1, 4), t)
    require((len(P), len(Qpart)) == (k+1, p), "Two collected supports at the table size")


def check_budgets():
    cases = [(22, 8, 2, 1, 5, 1600, 1120), (23, 8, 2, 1, 6, 1920, 1344),
             (24, 8, 2, 1, 7, 2240, 1568), (45, 16, 2, 1, 12, 10464, 8160),
             (50, 16, 2, 1, 17, 14824, 11560), (120, 16, 4, 2, 55, 123640, 100760),
             (180, 16, 6, 3, 83, 308096, 245016), (324, 32, 6, 3, 131, 1220920, 973592)]
    for m, n, L, a, k, count, previous in cases:
        S = n*(3*L-1)//2
        used, assigned, per_shelf = set(), set(), 0
        for block in layers(n, L):
            rows, modes = set(block.rows), set(block.frequencies)
            require(rows.isdisjoint(assigned) and modes.isdisjoint(used) and len(rows)*len(modes) == S
                    and max(modes) <= 2*S-1, "Mode budget and size per stage")
            for i in rows:
                require(all(sine_key(i*h, n) == 0 for h in used), "Earlier blocks invisible")
                signs = {sine_key(i*h, n)*s for h, s in zip(block.frequencies, block.signs)}
                require(len(signs) == 1 and abs(signs.pop()) == n//2, "Row signs factor exactly")
            used.update(modes)
            assigned.update(rows)
            per_shelf += len(rows)*model_count(a, len(modes))
        require(assigned == set(range(1, n)) and (k+1)+n*L == m, "Every well and exact split")
        require(best_parameters(m)[0] == count == 4*k*per_shelf == cycle_count(n, L, a, k),
                "Independent integer budget optimization and row count")
        prior = max(previous, filtered_bound(m), saddle_formula_ceiling(m), *comparators(m))
        require(best_translated(m) == previous and count > prior,
                "Translated, filtered, saddle, flat, row, separated and published comparators")
        minor = check_confluent_minor(n*L, n)
        check_table_support(n, L, a, k)
        print(f"m={m}, support ({k+1},{n*L}), {per_shelf} per shelf per quadrant, {count:,} cycles; "
              f"translated {previous:,}, best comparator {prior:,}; confluent minor {minor} modulo 65537.")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--field", action="store_true", help="Print the complete rational two-block sample in hex.")
    group.add_argument("--model", nargs=2, type=int, metavar=("A", "M"), help="Enclose one endpoint model.")
    args = parser.parse_args()
    if args.field:
        for name, part in zip(("P", "Q"), field_from_models(*model_sample())):
            print(name)
            for (x, y), c in sorted(part.items()):
                print(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}")
        return
    if args.model:
        a, M = args.model
        _, marks, kappa = certified_model(a, M)
        print(f"a={a}, M={M}: {len(marks)} strict marks, {model_count(a, M)} changes, "
              f"retained at kappa=2^-{kappa.denominator.bit_length()-1}.")
        return
    for n, L in ((2, 2), (4, 3), (8, 3)):
        check_visible_rank(n, L)
    check_moving_branches()
    check_second_order()
    for a, M in ((2, 11), (3, 8)):
        _, marks, kappa = certified_model(a, M)
        print(f"a={a}, M={M}: {len(marks)} rational marks, {model_count(a, M)} changes, "
              f"kappa=2^-{kappa.denominator.bit_length()-1}.")
    check_budgets()
    for m in range(4096, 24001):
        ceil_log = (m-1).bit_length()
        n = 1 << ((m//ceil_log).bit_length()-1)
        L = (m-1)//(2*n)
        p, a = n*L, L//2
        k = m-1-p
        require(1 <= a <= L//2 and n*ceil_log <= m < 2*n*ceil_log and k >= 1
                and (m-1)**2-4*k*p == (k-p)**2 < (2*n)**2, "All-budget allocation and rounding")
    print("All 19,905 exact allocations from 4,096 through 24,000 passed.")


if __name__ == "__main__":
    main()
