"""Run from the repository root to check the mixed-well and visible-frequency constructions."""

from argparse import ArgumentParser
from fractions import Fraction as Q
from math import comb, factorial, prod

from research.chebyshev_rows.verify import chebyshev
from research.exponent_variations.verify import inverse_and_determinant
from research.filtered_outlets.verify import at, evaluate
from research.mechanical_transfer.verify import sparse_history
from research.mixed_outlet.verify import polynomial_product
from research.reciprocal_wells.construction import best_parameters, cycle_count, model_count
from research.reciprocal_wells.endpoint_check import certified_model
from research.translated_wells.construction import affine
from research.verify_term_bounds import require
from research.mixed_multiwell.construction import (
    best_reciprocal, blocks, derivative, disk_model, half_gamma, linear_count, log_column,
    mixed_sample, quotient_degree, quotient_fit, reciprocal_count, reciprocal_sample,
)


def check_quotient():
    for n in (2, 4, 8, 16, 32):
        for L in range(2, 18):
            p, K, top = n*L, 2*n*L-n, quotient_degree(n, L)
            visible = {h for h in range(1, top+1) if h % n}
            modes = [h for _, freq in blocks(n, top) for h in freq]
            require(len(modes) == len(set(modes)) == 3*p-n-2*L and set(modes) == visible,
                    "Frequency allocation attains the joint visible dimension")
            require(sum(b % n != 0 for b in range(K+1, top+1)) == p-1 and (top+1) % n != 0,
                    "Exact last visible Taylor row")
            for a in (1, L//2):
                old_top = 3*p-n-1
                extra = 0
                for weight, new_modes in blocks(n, top):
                    old_size = sum(h <= old_top for h in new_modes)
                    extra += weight*(model_count(a, len(new_modes))-model_count(a, old_size))
                k = 3
                require(reciprocal_count(n, L, a, k)-cycle_count(n, L, a, k) == 4*k*extra,
                        "Exact count increment evaluated by added frequencies")
                direct = 0
                for e in range(n.bit_length()-1):
                    y = 2**e
                    size = sum((h & -h) == y for h in visible)
                    d = size-a
                    endpoint_changes = 2*((2*d-d//(a+1))//2)+2*(a-1)
                    direct += y*endpoint_changes
                require(reciprocal_count(n, L, a, k) == 4*k*direct,
                        "Independent dyadic classes and integer endpoint rounding")
    for n, L in ((2, 2), (2, 4), (4, 2), (8, 2)):
        p, K = n*L, 2*n*L-n
        rows = [b for b in range(K+1, quotient_degree(n, L)+1) if b % n]
        mat = [[log_column(j, b)*Q((-1)**(b-j-1)*factorial(b), factorial(j)*factorial(b-p-1))
                for j in range(2, p+1)] for b in rows]
        determinant = inverse_and_determinant(mat)[1]
        vandermonde = prod(v-u for i, v in enumerate(rows) for u in rows[:i])
        require(determinant == (-1)**comb(p-1, 2)*vandermonde, "Deleted-row Taylor determinant")

    field, (n, L, eta, target, poly, logs, blind) = reciprocal_sample()
    top = quotient_degree(n, L)
    require(len(target)-1 == top and all(blind.values()) and len(field[0]) == 4 and len(field[1]) == 8,
            "Enlarged highest frequency and collected (4,8) sample")
    ordinary, multiplier = affine(poly, 1, eta), affine(logs, 1, eta)
    log_series = [Q(0)]+[(-1)**(j-1)*eta**j/j for j in range(1, top+1)]
    expansion = polynomial_product(multiplier, log_series)
    for b in range(top+1):
        got = at(ordinary, b)+at(expansion, b)+sum(c*at(chebyshev(h), b) for h, c in blind.items())
        require(got == at(target, b), "Independent convolution of all finite-translation Taylor rows")
    z1, z2 = Q(3, 5), Q(4, 5)
    require(evaluate(chebyshev(n), z1) == evaluate(chebyshev(n), z2)
            and all(evaluate(chebyshev(h), z1) == evaluate(chebyshev(h), z2) for h in blind),
            "Auxiliary polynomials cancel on two distinct physical branches")
    # All ordinary and auxiliary columns have degree <= top. Beyond the fitted rows,
    # the absolute logarithmic-column coefficients decrease with row number.
    small_eta = Q(1, 2**128)
    _, small_logs, _ = quotient_fit(n, L, chebyshev(top), small_eta)
    first_tail = sum(abs(value)*abs(log_column(j, top+1)) for j, value in enumerate(small_logs))
    bound = 12*(top+1)**2 * small_eta**(top+1) * first_tail
    require(bound < Q(1, 2**80), "Function and first two derivative errors on [-1,1] for T21")
    # Decode the two ordinary components independently of the interpolation and its coordinates.
    P, Qpart = field
    lift = max(y for _, y in P)//2
    rates = {Q(y, lift)-1: -c for (_, y), c in P.items() if y != 2*lift}
    require(all(x == 1 for x, _ in P) and all(v == w for v, w in rates.items())
            and P[1, 2*lift] == sum(rates) and set(rates) == {Q(1, 16**i) for i in (1, 2, 3)},
            "Recovered the three damping rates and their sum from the first component")
    nodes = [Q(x, lift) for x, _ in sorted(Qpart)]
    tilts = [Q(y-1, lift)-1 for _, y in sorted(Qpart)]
    degree = max(x+y for part in field for x, y in part)
    require(all(abs(j-v) < Q(1, 4) for j, v in enumerate(nodes, 1))
            and all(abs(w) < Q(1, 2) for w in tilts)
            and degree == max(1+2*lift, *(1+lift*(v+1+w) for v, w in zip(nodes, tilts)))
            and degree.bit_length() == 14778, "Recovered the nodes, tilts and exact lifted degree")
    for a, M in ((1, 6), (1, 11), (2, 12)):
        _, marks, _ = certified_model(a, M)
        count = sum(left[-1] != right[-1] for left, right in zip(marks, marks[1:]))
        require(count == model_count(a, M), "Regularized endpoint enclosures at tested sizes")
    print("Deleted-row determinants, 22 Taylor rows, uniform C^2 error below 2^-80 for T21, "
          "a 14,778-bit collected degree, and three regularized endpoint models passed.")


def check_fixed_field():
    tau = Q(1, 2**40)
    P, Qpart = mixed_sample("six")
    require(P == {(0, 5): 6*tau, (0, 3): 2-12*tau, (0, 1): 21*tau/4-2}
            and Qpart == {(1, 0): -1, (0, 4): tau/2, (0, 2): -tau},
            "Full coefficient readback for the certified six-term field")

    # One full turn of a product of powers of sin(theta) and cos(theta), divided by 2*pi.
    def mean(sine_power, cosine_power):
        if sine_power % 2 or cosine_power % 2:
            return Q(0)
        i, j = sine_power//2, cosine_power//2
        return half_gamma(i)*half_gamma(j)/factorial(i+j)

    def product_mean(left, right):
        return sum(a*b*mean(s+t, c+d) for s, c, a in left for t, d, b in right)

    # The first radial correction q1 is the integral of A from 0 to theta.
    # The quadratic coefficient is the integral of A_r*q1 + A*C/r.
    for r in (Q(1, 2), Q(2, 3), Q(7, 10), Q(3, 4), Q(4, 5)):
        A = [(2, 1, 3*r*r), (0, 1, Q(-3, 8)), (3, 0, -r*r/2), (1, 0, Q(1, 2))]
        C = [(3, 0, 3*r*r), (1, 0, Q(-3, 8)), (2, 1, r*r/2), (0, 1, Q(-1, 2))]
        Ar = [(2, 1, 6*r), (3, 0, -r)]
        q1 = [(3, 0, r*r), (1, 0, Q(-3, 8)), (0, 0, Q(1, 2)-r*r/3),
              (0, 1, r*r/2-Q(1, 2)), (0, 3, -r*r/6)]
        require(sum(a*mean(s, c) for s, c, a in A) == 0, "First radial coefficient vanishes")
        require(product_mean(Ar, q1)+product_mean(A, C)/r == -Q(3, 16)*r*(2*r*r-1),
                "Direct polar expansion of the signed quadratic return coefficient")
    radius = Q(1, 2**10)
    require(70*radius < Q(1, 8) and 15*radius < Q(1, 2), "Complex radial continuation")
    error = 2*tau/radius**3
    require(error < min(Q(3, 36), Q(27, 256)) and 16*error < Q(15, 8),
            "Cauchy errors below both signs and the uniform slope at the specified tau")
    print("The degree-five six-term field at tau=2^-40 passed the polar signs and slope bounds.")


def check_branches():
    # On either Duffing branch z=y^2-1 the normalized mixed moment is exact.
    zero = Q(1, 2)
    moment = lambda t: t*(3*t/4-Q(3, 8))
    require(moment(zero) == 0 and moment(Q(1, 4)) < 0 < moment(Q(3, 4))
            and 3*zero/2-Q(3, 8) == Q(3, 8), "Two-well analytic period zero and slope")
    # When the two baseline functions equal their potentials, translations and quadratic
    # perturbations keep the deformed disk an ellipse. Its xy moment is area times cx*cy.
    for lam, eta in ((Q(1, 17), Q(-1, 31)), (Q(-1, 41), Q(1, 13)), (Q(0), Q(1, 7))):
        a, b, c, d = 2, -3, 5, 7
        cx, cy = -lam*a/(1+lam*c), -eta*b/(1+eta*d)
        require(cx*cy == lam*eta*(a+c*cx)*(b+d*cy), "Exact finite mixed area on a moving ellipse")
    # B_x=B_y=T4 and U=V=1-T4^2 at the two opposite monotone branches.
    J = chebyshev(4)
    xx = (Q(3, 5), Q(4, 5))
    differences = []
    for shape in ([1, 2, 3], [0, -3, 0, 1]):
        moving, primitive = [], []
        for x in xx:
            value, slope = evaluate(J, x), evaluate(derivative(J), x)
            energy_slope = -2*value*slope
            velocity = -evaluate(shape, x)/energy_slope
            moving.append(slope*velocity)
            primitive.append(evaluate(shape, x)/(2*value))
        require(moving == primitive and moving[0] != moving[1], "Nonzero signed shape difference")
        differences.append(moving[1]-moving[0])
    require(prod(differences) != 0, "Both deformed slice factors survive")
    for n in (2, 4, 8, 16, 32, 64):
        earlier = []
        for k, (b, freq) in enumerate(blocks(n, 3*n+2)):
            rows = [i for i in range(1, 2*n) if i % (2**(k+1)) == 2**k]
            require(len(rows) == 2*b, "Vertical well multiplicity")
            for i in rows:
                require(all(h*i % n == 0 for h in earlier), "Every earlier block vanishes on this row")
                residues = [(h*i) % (2*n) for h in freq]
                require(residues[0] in (n//2, 3*n//2) and
                        residues == [(residues[0]+j*n) % (2*n) for j in range(len(freq))],
                        "Common diagonal row signs")
            earlier.extend(freq)
    print("Exact finite ellipse moment, two nonzero moving branches, and dyadic well signs passed.")


def packet_moment(exponents, s):
    moment = {}
    for a in exponents:
        for j in range(s+1):
            power = a+j+1
            moment[power] = moment.get(power, 0)+Q((-1)**j*comb(s, j), factorial(power))
    return moment


def model_roots():
    # Direct physical disk integral with gamma factors versus the derivative formula.
    exponents = (1, 25, 81)
    marks = {1: (1, 4, 14, 27, 53, 83), 3: (1, 2, 5, 10, 16, 20, 29, 39, 55, 70, 85, 102)}
    for s, points in marks.items():
        source = disk_model(s+1, True)
        direct = {}
        for a in exponents:
            for j, c in enumerate(source):
                order = a+j+1
                value = (1/half_gamma(a))*(2*j+1)*c*half_gamma(a)*half_gamma(j)/factorial(order)
                direct[order] = direct.get(order, 0)+value
        moment = packet_moment(exponents, s)
        require(moment == direct, "Collected product coefficients integrated on the disk")
        for k, t in enumerate(points):
            require((-1)**k*sum(value*t**power for power, value in moment.items()) > 0,
                    "Strict rational sign at the specified energy")
        coeff = [v for _, v in sorted(moment.items()) if v]
        require(sum((u < 0) != (v < 0) for u, v in zip(coeff, coeff[1:])) == len(points)-1,
                "Ordered marks exhaust the Descartes sign changes")
    for a in range(8):
        for b in range(8):
            local = [disk_model(s+1, True) for s in (a, b)]
            normalized = [[x*(2*j+1)*half_gamma(j) for j, x in enumerate(poly)] for poly in local]
            conv = polynomial_product(*normalized)
            require(conv == [(-1)**j*comb(a+b, j) for j in range(a+b+1)], "Sharp dense disk polynomial")
    print("All 5 and 11 simple packet roots certified by rational marks and Descartes. "
          "Dense disk products checked through bidegree (7,7).")


def best_linear_at(m):
    best, params = 0, None
    for exponent in range(1, m.bit_length()):
        n = 2**exponent
        for d in range(2*n, (m+1)//2+1):
            wells = m-d-n
            if wells < 1 or 2*wells-1 > m:
                continue
            count = linear_count(n, d, wells)
            if count > best:
                best, params = count, (n, d, wells)
    return best, params


def old_linear_cap(m):
    best = 0
    for d in range(2, (m+1)//2+1):
        vertical = 2*d-d % 2-1
        wells = min((m+1)//2, m-vertical)
        if wells > 0:
            best = max(best, sparse_history(d, wells)[2])
    return best


def check_budgets():
    for kind, sizes, degree in (("linear", (8, 9), 15), ("grid", (8, 14), 15), ("packets", (8, 8), 162)):
        field = mixed_sample(kind)
        require(tuple(map(len, field)) == sizes and max(x+y for part in field for x, y in part) == degree,
                "Collected both components and degree of the chosen mixed sample")
    for m in (63, 95, 128, 256, 512):
        current, (n, d, wells) = best_linear_at(m)
        old = old_linear_cap(m)
        require(current > old, "Improvement over the old mechanical formula at both resource caps")
        print(f"m={m}, (n,d,wells)=({n},{d},{wells}), degree {2*max(d,wells)-1}, "
              f"supports ({d},{n+wells}), {current:,} mixed cycles. Previous linear history {old:,}.")
    for m in range(128, 32001):
        ceiling_log = (m-1).bit_length()
        n = 1 << ((m//ceiling_log).bit_length()-1)
        # Using the ceiling may lose one dyadic step. Test m >= 2*n*log2(m)
        # exactly in the narrow range where that next step can fit.
        if m > 2*n*(ceiling_log-1) and (1 << m) >= m**(2*n):
            n *= 2
        d, wells = (m-n)//2, (m-n+1)//2
        require(d >= 2*n and d+wells+n == m and 2*max(d, wells)-1 <= m, "All-budget linear split")
        packet_d = 2*n*((m-n-1)//(4*n))
        p = m-n-1-packet_d
        require(packet_d >= 2*n and packet_d % (2*n) == 0 and p > 0 and packet_d+p+n+1 == m,
                "All-budget separated-exponent split")
    for m, expected in ((22, 1728), (23, 1968), (24, 2296), (45, 10880), (50, 14960),
                        (120, 124960), (180, 314072), (324, 1230352)):
        new, params = best_reciprocal(m)
        old, _ = best_parameters(m)
        require(new == expected and new > old, "Optimized reciprocal count exceeds the comparison count")
        print(f"m={m}, reciprocal {params}, {new:,} cycles, comparison {old:,}, gain {new-old:,}.")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--budget", type=int, help="Optimize the two new formulas at an exact budget.")
    group.add_argument("--field", choices=("six", "linear", "grid", "packets", "reciprocal"),
                       help="Emit both fully collected supports and rational coefficients in hexadecimal.")
    args = parser.parse_args()
    if args.budget is not None:
        if args.budget < 1:
            parser.error("The term budget must be positive.")
        print("Linear degree", best_linear_at(args.budget))
        print("Reciprocal", best_reciprocal(args.budget))
        print("Comparison count", best_parameters(args.budget))
        return
    if args.field:
        field = reciprocal_sample()[0] if args.field == "reciprocal" else mixed_sample(args.field)
        for name, component in zip(("P", "Q"), field):
            print(name)
            for (x, y), c in sorted(component.items()):
                print(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}")
        return
    check_quotient()
    check_fixed_field()
    check_branches()
    model_roots()
    check_budgets()


if __name__ == "__main__":
    main()
