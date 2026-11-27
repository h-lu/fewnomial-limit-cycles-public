"""Ordinary supports, exact expansions and rank checks for the mechanical transfer."""

from argparse import ArgumentParser
from fractions import Fraction as Q
from hashlib import sha256
from itertools import combinations, combinations_with_replacement
from math import comb, factorial, prod
from pathlib import Path
from random import Random

from research.chebyshev_rows.verify import chebyshev
from research.mixed_outlet.verify import polynomial_product
from research.verify_term_bounds import require


def prefixes(n):
    return [n >> i for i in reversed(range(n.bit_length()))]


def birth_modes(N, s, A):
    return [(p, q) for q in range(s//A) for p in range(N-A*(2*q+1))]


def downward(support):
    """Taylor support at a generic two-coordinate translation."""
    height = {}
    for x, y in support:
        for u in range(x+1):
            height[u] = max(height.get(u, -1), y)
    return {(u, v) for u, top in height.items() for v in range(top+1)}


def closed_budget(n):
    """Binary digit formula for the frozen field, with and without the latest axis sources."""
    s = n//2
    odd = comb(n, 2)+1-n.bit_count()
    old_odd = comb(s, 2)+1-s.bit_count()
    axis_overlap = 2*s-2-2*(s % 2) if s >= 2 else 0
    copied_and_potential = 4*old_odd+n-axis_overlap
    return odd+copied_and_potential, copied_and_potential


def mechanical_budget(n):
    """Generic collected first component, with the second obtained by transposition."""
    if n < 2:
        raise ValueError("Require n >= 2.")
    form, cycles = set(), 0
    levels = prefixes(n)
    for i, (s, N) in enumerate(zip(levels, levels[1:]), 1):
        row_degree = {}
        K = 0
        for k in range(i):
            A = 2**k
            modes = birth_modes(N, s, A)
            d, e = divmod(s, A)
            require(len(modes) == d*(N-A*d) and A*len(modes) == (s-e)*(N-s+e),
                    "Birth modes and copy-weighted rank")
            K += 4*A*(len(modes)-1)
            for p, q in modes:
                row_degree[p] = max(row_degree.get(p, -1), A*(2*q+1)-1)
        newest = {(2*p+1, 2*j+1) for p, q in birth_modes(N, s, 1) for j in range(2*q+1)}
        require(len(newest) == N*(N-1)//2-(N-1)//2, "Newest-birth support in one axis source")
        odd = {(2*p+1, 2*j+1) for p, top in row_degree.items() for j in range(top+1)}
        old = {(2*u, 2*v+1) for u, v in downward(form)}
        require(odd.isdisjoint(old) and newest <= odd, "Separate parity sectors")
        require(downward(odd | old) == downward(odd) and len(downward(odd)) == 4*len(odd),
                "New odd rows dominate all old copied rows and tile the entire downward closure")
        form = odd | old
        cycles = 4*cycles+K
    field = form | {(0, 2*j+1) for j in range(n)}
    require(max(x+y for x, y in field) == 2*n-1, "Field degree")
    return field, cycles


def shifted(poly, b):
    """Coefficients of poly(X-b) by Horner multiplication in ordinary powers."""
    poly = list(poly)
    out = [poly.pop()]
    for c in reversed(poly):
        out = [c-b*out[0]] + [lo-b*hi for lo, hi in zip(out, out[1:])] + [out[-1]]
    return out


def fold_row(row, b):
    values = shifted(row, b)
    return [item for c in values for item in (0, 2*c)]


def add_entry(values, index, value):
    values[index] = values.get(index, 0)+value


def copy_form(form, a, b):
    copied = {}
    for (u, v), coeff in form.items():
        for j in range(u+1):
            for k in range(v+1):
                contribution = 2*coeff*comb(u, j)*(-a)**(u-j)*comb(v, k)*(-b)**(v-k)
                add_entry(copied, (2*j, 2*k+1), contribution)
    return copied


def sample_aux(levels, label, folds, rng):
    """Rebuild all previous births for this future label, with integer sample weights."""
    s, N = levels[label-1:label+1]
    rows = {}
    for j in range(label):
        if j:
            rows = {p: fold_row(row, folds[j-1]) for p, row in rows.items()}
        A = 2**(label-1-j)
        for p, q in birth_modes(N, s, A):
            row = rows.setdefault(p, [])
            if len(row) <= 2*q:
                row.extend([0]*(2*q+1-len(row)))
            row[2*q] += rng.randrange(1, 97)
    return rows


def fold_potential(poly, shift, size):
    out = [u for c in shifted(poly, shift) for u in (c, 0)][:-1]
    if size % 2:
        out.extend([0, 3])
    return out


def expand_one_axis(levels, x_folds, y_folds, rng, perturbation_sign):
    form, x_potential, y_potential = {}, [0, 0, 1], [0, 0, 1]
    for i, (a, b) in enumerate(zip(x_folds, y_folds), 1):
        form = copy_form(form, a, b)
        scale_squared = 2*sum(j*c*a**(j-1) for j, c in enumerate(x_potential) if j)
        require(scale_squared > 0, "Positive kinetic scale")
        for p, row in sample_aux(levels, i, y_folds, rng).items():
            for j, coeff in enumerate(shifted(row, b)):
                # Choose the allowed axis amplitude 1/c_a to keep all coefficients integral.
                add_entry(form, (2*p+1, 2*j+1), 2*coeff*scale_squared**p)
        x_potential = fold_potential(x_potential, a, levels[i])
        y_potential = fold_potential(y_potential, b, levels[i])
    field = {key: perturbation_sign*13*value for key, value in form.items()}
    for j, c in enumerate(y_potential[1:], 1):
        add_entry(field, (0, j-1), -j*c)
    return {index: value for index, value in field.items() if value}


def sample_field(n):
    levels = prefixes(n)
    x_folds = [113+79*i for i in range(len(levels)-1)]
    y_folds = [157+73*i for i in range(len(levels)-1)]
    P = expand_one_axis(levels, x_folds, y_folds, Random(7000+n), -1)
    raw = expand_one_axis(levels, y_folds, x_folds, Random(9000+n), 1)
    Qpart = {(y, x): value for (x, y), value in raw.items()}
    return P, Qpart


def matching(columns):
    """Rank of the even Taylor rows on an ordered subset of physical columns."""
    rank = 0
    for j in sorted(columns):
        if j >= 2*rank:
            rank += 1
    return rank


def modular_rank(rows, prime=65537):
    rows = [[value % prime for value in row] for row in rows]
    rank = 0
    for j in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][j]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][j], -1, prime)
        for i in range(rank+1, len(rows)):
            factor = rows[i][j]*inv
            rows[i] = [(u-factor*v) % prime for u, v in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def check_pascal():
    for mask in range(1, 1 << 11):
        cols = [j for j in range(11) if mask & (1 << j)]
        rows = [[comb(j, 2*q)*7**max(0, j-2*q) for j in cols] for q in range(6)]
        require(modular_rank(rows) == matching(cols), "Collected outer rank from two calculations")
    # Every eligible small Pascal minor has the predicted strict or zero sign.
    for size in range(1, 4):
        choices = list(combinations(range(8), size))
        for lo, hi in combinations_with_replacement(choices, 2):
            for rows, cols in ((lo, hi), (hi, lo)):
                mat = [[comb(c, r) for c in cols] for r in rows]
                if size == 1:
                    det = mat[0][0]
                elif size == 2:
                    det = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
                else:
                    a, b, c = mat
                    det = sum(a[i]*(b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3]) for i in range(3))
                require(det >= 0 and (det > 0) == all(r <= c for r, c in zip(rows, cols)),
                        "Pascal minor sign and matching criterion")
    print("All 2,047 nonempty column supports through degree ten and all small Pascal minor signs passed.")


def check_compressed_birth():
    """Reconstruct a model birth exactly from squared even physical columns on each axis."""
    for N in range(4, 34):
        s, eta = divmod(N, 2)
        b = 7
        physical, wanted = {}, {}
        for p in range(N-1):
            width = (N-2-p)//2+1
            target = [(-1)**q * (p+q+1) for q in range(width)]
            coeff = target.copy()
            for q in reversed(range(width)):
                for j in range(q):
                    coeff[j] -= comb(2*q, 2*j)*b**(2*q-2*j)*coeff[q]
            poly = [0]*(2*width-1)
            for q, value in enumerate(coeff):
                poly[2*q] = value
                physical[2*p+1, 4*q+1] = 2*value
            even_after_shift = shifted(poly, -b)[::2]
            require(even_after_shift == target, "Sparse columns reproduce the entire even model row")
            for q, value in enumerate(even_after_shift):
                wanted[p, q] = value
        require(set(wanted) == set(birth_modes(N, s, 1)) and len(wanted) == s*(s+eta),
                "The compressed physical support carries the entire newest birth")
        require(all(physical.values()) and len(physical) == len(wanted), "Occupied compressed source slots")
        # Unit central force potentials (Y-b)^(2s) plus the odd-prefix padding.
        for j in range(1, 2*s+1):
            physical[0, 2*j-1] = -2*j*comb(2*s, j)*(-b)**(2*s-j)
        if eta:
            physical[0, 2*N-1] = -2*N
        require(len(physical) == N+s*(s+eta) and max(u+v for u, v in physical) == 2*N-1,
                "Compressed source plus pure Hamiltonian after collection")
    print("Inverse even translations, newest modes, and collected field sizes passed for N=4..33.")
    for N in (4, 5, 6, 7):
        s = N//2
        exp = sorted(s*(2*p+1)+2*q+1 for p, q in birth_modes(N, s, 1))
        nodes = [2**e for e in exp]
        weights = [Q(1, u*prod(u-v for v in nodes if v != u)) for u in nodes]
        for j in range(1, len(nodes)):
            t = 2**j
            require(sum(c*t**e for c, e in zip(weights, exp)) == 0, "Geometric marked period zero")
            slope = sum(e*c*t**(e-1) for c, e in zip(weights, exp))
            require(slope != 0, "Nonzero normalized mechanical slope at the marked zero")
    print("Homogeneous newest-birth simple zeros and their slopes verified at N=4,5,6,7.")


def check_folds():
    # A folded primitive E(Y) with all roots and the chosen small levels on Y>0.
    E = [-20, 1]
    for A in (1, 2, 4):
        if A > 1:
            E = polynomial_product(E, E)
            E[0] -= 3 if A == 2 else 1
        derivative = [j*E[j] for j in range(1, len(E))]
        for d in (1, 2, 3, 4):
            coeff = derivative
            for j in range(1, d):
                factor = polynomial_product(E, E)
                factor[0] -= Q(j, 16)**2
                coeff = polynomial_product(coeff, factor)
            require(len(coeff) == A*(2*d-1) and all(coeff), "Sharp copied row support")
        for width in (1, 2, 3):
            coeff = [Q(1)]
            for _ in range(2*width-1):
                coeff = polynomial_product(coeff, E)
            require(sum(bool(v) for v in coeff) == A*(2*width-1)+1, "Sharp odd kinetic row support")
    for degree in range(1, 32):
        for r in range(1, degree+2):
            u, v = degree-r+1, r-1
            poly = [0]*u+[comb(v, j)*3**(v-j) for j in range(v+1)]
            require(sum(map(bool, poly)) == r and sum(map(bool, shifted(poly, 3))) == degree+2-r,
                    "Equality for shift uncertainty")
    print("Folded row support and Taylor-shift uncertainty equality examples passed.")


def check_reciprocal_copy():
    from research.reciprocal_wells.verify import field_from_models, model_sample

    P, Qpart = field_from_models(*model_sample())
    k, modes = len(P)-1, len(Qpart)
    lift = max(v for _, v in P)//2
    powers = sorted(Qpart)
    e, v = powers[-1]
    force_minimum = (e+2-modes)*(v+1)
    # Count the union of rectangles in width strips, without expanding the large degrees.
    right_edge = -1
    force_generic = 0
    for j, (edge, _) in enumerate(powers):
        top = max(height for _, height in powers[j:])
        force_generic += (edge-right_edge)*(top+1)
        right_edge = edge
    require((len(P), modes) == (4, 8) and all(i == 1 for i, _ in P), "Four-by-eight field input")
    require(4*e > (4*modes-1)*lift and 2*v > lift, "Physical bounds on the last exponents")
    require(2*(2*lift+1-k) <= 2*(2*lift+1) and 0 < force_minimum <= force_generic,
            "Copied component lower estimates fit their generic envelopes")
    print(f"Specified (4,8) sample, lift has {lift.bit_length()} bits. "
          f"The copied second component needs at least 2^{force_minimum.bit_length()-1} terms.")


def sparse_history(d, wells):
    if d < 2 or wells < 1:
        raise ValueError("Require vertical half-degree d >= 2 and at least one horizontal well.")
    levels = prefixes(d)
    count = 0
    for j, size in enumerate(levels[1:], 1):
        count *= 2
        if j < len(levels)-1:
            count += size-1
    return (2*d-(d % 2)-1, wells), 2*max(d, wells)-1, wells*count


def sparse_sample(d, wells):
    levels = prefixes(d)
    V, row = [0, 0, 1], [2]
    for j, size in enumerate(levels[1:], 1):
        b = 127+17*j
        V = [u for c in shifted(V, b) for u in (c, 0)][:-1]
        if size % 2:
            V.extend([0, 1])
        row = fold_row(row, b)
        if j < len(levels)-1:
            row.extend([0]*max(0, 2*size-1-len(row)))
            for q in range(size):
                row[2*q] += q+5
    T = chebyshev(wells)
    Qpoly = polynomial_product(T, [j*T[j] for j in range(1, len(T))])
    P = {(0, j-1): j*c for j, c in enumerate(V) if j and c}
    P.update({(1, j): -c for j, c in enumerate(row) if c})
    return P, {(j, 0): -v for j, v in enumerate(Qpoly) if v}


def check_sparse():
    for d in range(2, 33):
        for wells in (3, d, 2*d):
            support, degree, count = sparse_history(d, wells)
            P, Qpart = sparse_sample(d, wells)
            require((len(P), len(Qpart)) == support and max(x+y for x, y in P.keys() | Qpart.keys()) == degree,
                    "Complete collected sparse sample")
            t = d.bit_length()-1
            explicit = wells*(t*d-sum(d % (1 << l)+(1 << l) for l in range(1, t+1)))
            require(count == explicit, "One-coordinate history by levels and recurrence")
    for m in range(64, 20001):
        d = m//4
        wells = m-2*d+(d % 2)+1
        support, degree, _ = sparse_history(d, wells)
        require(sum(support) == m and degree <= m+6, "All-budget sparse split and degree")
    print("Both sparse components expanded for d=2..32. All exact splits m=64..20,000 passed.")


def mixed_sample(p, q):
    """Expand the two-component product example and its circular moment exactly."""
    K, T = p*q, 16*p*q
    half_gamma = lambda a: Q(factorial(2*a), 4**a*factorial(a))
    P, Qpart = {(0, 1): Q(1)}, {(1, 0): Q(-1)}
    delta, mu = Q(1, 17), Q(1, 19)
    for i in range(p):
        a = i*T
        Qpart[2*a, 0] = -mu*(-1)**i/half_gamma(a)
    for j in range(q):
        b = p*j*T
        Qpart[1, 2*b+1] = -delta*(-1)**(p*j)/((2*b+1)*half_gamma(b))
    moment = {}
    for (e, v), c in Qpart.items():
        if v or e % 2:
            continue
        a = e//2
        for (u, f), g in Qpart.items():
            if u != 1 or not f:
                continue
            b = (f-1)//2
            order = a+b+1
            moment[order] = c*g*f*half_gamma(a)*half_gamma(b)/(mu*delta*factorial(order))
    require((len(P), len(Qpart)) == (1, p+q+1) and
            max(x+y for x, y in Qpart) == max(2*(p-1)*T, 2*p*(q-1)*T+2),
            "Sparse product field has its stated separate supports and exact degree")
    require(moment == {k*T+1: Q((-1)**k, factorial(k*T+1)) for k in range(K)},
            "Physical coefficients reproduce every signed factorial moment")
    for k in range(K):
        x = k*T+1
        value = sum(c*x**order for order, c in moment.items())
        require((-1)**k*value > 0, "One rational energy mark per product pair")


def check_frozen(folder):
    entries = dict((name, digest) for digest, name in
                   (line.split(maxsplit=1) for line in (folder/"SHA256SUMS").read_text().splitlines()))
    expected = {"main.pdf": "5eea3b245a109435b4298f76bede1c91b7c23179857af717b926b6e9f0c7f984",
                "mechanical-proof.tex": "7b87652b0e946c5e595568affb48564a521383bf448ff0ad1f74a659d3665c1e"}
    names = ["main.pdf", "main.tex", "prelim.tex", "mechanical-proof.tex", "counting.tex",
             "replication.tex", "renewal-conjecture.tex"]
    for name in names:
        got = sha256((folder/name).read_bytes()).hexdigest()
        require(got == entries[name] == expected.get(name, got), "Read-only frozen source identity")
    print("Frozen PDF and all six TeX files agree with their manifest.")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--budget", type=int, help="Compute both generic supports and the frozen count at N.")
    group.add_argument("--algebra-only", action="store_true",
                       help="run the algebraic checks without the separate October 24 source archive")
    group.add_argument("--field", type=int, metavar="N", help="Print both collected integer support samples.")
    group.add_argument("--frozen-source", type=Path, metavar="DIR",
                       help="also verify the October source manifest and files in DIR")
    args = parser.parse_args()
    if args.budget is not None:
        support, count = mechanical_budget(args.budget)
        print(f"D={2*args.budget-1}, supports ({len(support)},{len(support)}), selected cycles {count}.")
        return
    if args.field is not None:
        if args.field < 2:
            parser.error("The sampled half-degree N must be at least two.")
        for name, component in zip(("P", "Q"), sample_field(args.field)):
            print(name)
            for (x, y), coefficient in sorted(component.items()):
                print(f"{x:x} {y:x} {coefficient:x}")
        return
    if args.frozen_source is None:
        print("Algebraic checks only. The external October 24 manuscript digest is outside this run.")
    else:
        check_frozen(args.frozen_source)
    for n in range(2, 49):
        support, count = mechanical_budget(n)
        closed, copied = closed_budget(n)
        require(closed == len(support) and copied == sum(x % 2 == 0 for x, _ in support),
                "Binary closed formula against both enumerated sectors")
        if n <= 23:
            P, Qpart = sample_field(n)
            require(set(P) == support and set(Qpart) == {(y, x) for x, y in support},
                    "Full ordinary expansion of two independent coefficient samples")
        if not n & (n-1):
            r = n.bit_length()-1
            require(2*len(support) == 2*n*n-3*n+4
                    and 3*count == 3*n*n*r*(r+1)//2-8*n*n+12*n-4, "Both dyadic closed forms")
            require(2*copied == n*n-2*n+4, "Dyadic copied field after final sources are removed")
        if n % 2 == 0:
            require(2*copied <= n*n, "All even endpoints with last global birth removed")
        if n in (4, 7, 8, 13, 16, 24, 32):
            print(f"N={n:2}, D={2*n-1:2}, supports ({len(support)},{len(support)}), L={count}.")
    check_pascal()
    check_compressed_birth()
    check_folds()
    check_reciprocal_copy()
    check_sparse()
    for p, q in ((2, 2), (3, 4)):
        mixed_sample(p, q)
    print("Bilinear circular moments and all marks verified at (p,q)=(2,2),(3,4).")


if __name__ == "__main__":
    main()
