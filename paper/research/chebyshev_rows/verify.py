"""Exact series, row filtration, support and budget checks for the one-axis grid.

Run python3 -B -m research.chebyshev_rows.verify from the repository root.
"""

from argparse import ArgumentParser
from fractions import Fraction as Q
from math import comb, factorial, prod

from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.mixed_outlet.verify import polynomial_product
from research.verify_term_bounds import require


def layer(n, k):
    modulus = 3**k
    return [j*n//modulus for j in range(1, modulus) if j%3]


def cycle_count(n, r, vertical):
    return vertical * (4*(n//3)*r - n + 1)


def budget(m):
    if m < 7:
        raise ValueError("This allocation requires m >= 7.")
    n, r = 3, 1
    while 8*(3*n) <= 3*(m+1):
        n *= 3
        r += 1
    vertical = m+1-2*n
    return n, r, vertical, cycle_count(n, r, vertical)


def jets(frequency, n, vertical, length):
    """Taylor series divided by the disk area, with h equal to Hamiltonian energy."""
    def products(x):
        result = [Q(1)]
        for i in range(1, length):
            result.append(result[-1] * ((2*i-1)**2-x*x)/i)
        return result

    horizontal = products(Q(2*frequency, n))
    passive = products(Q(1, vertical))
    return [sum(horizontal[p]*passive[l-p] for p in range(l+1))
            / (2**l * factorial(l+1)) for l in range(length)]


def sin_sign(m):
    return (0, 1, 1, 0, -1, -1)[m%6]


def check_layers():
    for r in range(1, 6):
        n = 3**r
        frequencies, checked_rows = [], []
        for k in range(1, r+1):
            divis = 3**(k-1)
            rows = [i for i in range(n) if (2*i+1)%divis == 0 and (2*i+1)%(3*divis)]
            block = layer(n, k)
            require(len(rows) == 2*n//(3*divis) and len(block) == 2*divis, "Layer sizes")
            for i in rows:
                require(all(s*(2*i+1)%n == 0 for s in frequencies), "Lower blocks vanish")
                unit = (2*i+1)//divis
                for s in block:
                    index = 3*s*(2*i+1)//n
                    m = s//(n//(3*divis))
                    require(3*s*(2*i+1) == n*index
                            and sin_sign(index) == sin_sign(unit)*sin_sign(m) != 0,
                            "Diagonal sine factorization")
            frequencies.extend(block)
            checked_rows.extend(rows)
        require(sorted(frequencies) == list(range(1, n)), "Frequency partition")
        require(sorted(checked_rows + [(n-1)//2]) == list(range(n)), "Center partition")


def check_rank_and_unfolding():
    for n, r, vertical in ((3, 1, 1), (9, 2, 5), (27, 3, 54)):
        for k in range(1, r+1):
            s = layer(n, k)
            d = len(s)
            matrix = list(map(list, zip(*(jets(j, n, vertical, d) for j in s))))
            inverse, det = inverse_and_determinant(matrix)
            nodes = [Q(2*j, n)**2 for j in s]
            vandermonde = prod(v-u for i, v in enumerate(nodes) for u in nodes[:i])
            leading = prod(Q((-1)**p, 2**p*factorial(p)*factorial(p+1)) for p in range(d))
            require(det == leading*vandermonde != 0, "Vandermonde kernel determinant")
            rho = Q(1, 64)
            polynomial = [Q(1)]
            for root in range(1, d):
                polynomial = polynomial_product(polynomial, [-root*rho, 1])
            control = matvec(inverse, polynomial)
            require(matvec(matrix, control) == polynomial, "Exact unfolding coefficients")


def chebyshev(d):
    prev, curr = [1], [0, 1]
    for _ in range(1, d):
        nxt = [0] + [2*x for x in curr]
        for j, x in enumerate(prev):
            nxt[j] -= x
        prev, curr = curr, nxt
    return curr


def restoring(d, j):
    return d * (-1)**(d-1-j) * 4**j * comb(d+j, d-1-j)


def primitive(s, j):
    return Q((-1)**(s-j) * 2**(2*j-1) * comb(s+j, s-j), s+j)


def expanded(n, vertical, modes, shift, perturbation):
    """Collected monomials for the supplied rational parameters, including any cancellations."""
    p = {(0, 2*j+1): restoring(vertical, j) for j in range(vertical)}
    for j in range(1, n):
        coefficient = shift + sum(modes[s-1]*primitive(s, j) for s in range(j, n))
        if coefficient and perturbation:
            p[2*j, 0] = perturbation*coefficient
    q = {(2*j+1, 0): -restoring(n, j) for j in range(n)}
    return p, q


def check_support():
    for d in range(1, 31):
        t = chebyshev(d)
        product = polynomial_product(t, [j*t[j] for j in range(1, d+1)])
        require(all(product[2*j] == 0 and product[2*j+1] == restoring(d, j)
                    for j in range(d)), "Restoring force by independent Chebyshev recurrence")
        t = chebyshev(2*d)
        require(all(primitive(d, j) == Q(t[2*j], 2*d) for j in range(1, d+1)),
                "Perturbation coefficients by independent Chebyshev recurrence")
    for n, vertical in ((3, 3), (9, 8), (27, 54)):
        modes = [Q((-1)**s, s+1) for s in range(1, n)]
        mu, shift = Q(1, 97), Q(1, 103)
        p, q = expanded(n, vertical, modes, shift, mu)
        require((len(p), len(q)) == (vertical+n-1, n), "Collected supports in sample expansion")
        for j in range(1, n):
            separate = sum(c*Q(chebyshev(2*s)[2*j], 2*s)
                           for s, c in enumerate(modes, 1) if s >= j)
            require(p[2*j, 0] == mu*(separate+shift), "Full collected x-axis disturbance")


def check_two_layers_at_fixed_shape():
    """Simultaneous model signs and derivatives at n=9, N=5; the final mu is existential."""
    n, vertical = 9, 5
    rho, zeta, shift = Q(1, 2**96), Q(1, 2**1024), Q(1, 2**8192)
    layer_norm = {}
    modes = [Q(0)] * (n-1)
    for k in (1, 2):
        block = layer(n, k)
        d = len(block)
        matrix = list(map(list, zip(*(jets(s, n, vertical, d) for s in block))))
        polynomial = [Q(1)]
        for root in range(1, d):
            polynomial = polynomial_product(polynomial, [-root*rho, 1])
        coefficients = matvec(inverse_and_determinant(matrix)[0], polynomial)
        layer_norm[k] = sum(map(abs, coefficients))
        for s, v in zip(block, coefficients):
            modes[s-1] = zeta**k * sin_sign(s//(n//3**k)) * v

    # Reexpress the support-filling polynomial in the same primitive basis.
    power_matrix = [[primitive(s, j) if s >= j else 0 for s in range(1, n)]
                    for j in range(1, n)]
    shift_modes = matvec(inverse_and_determinant(power_matrix)[0], [Q(1)]*(n-1))
    require(matvec(power_matrix, shift_modes) == [1]*(n-1), "Shift in the Chebyshev basis")
    for k in (1, 2):
        d = len(layer(n, k))
        # |[h^l]K| <= 2(l+1)2^l. On 0 <= z <= 6 at h=rho*z this
        # bounds both the scaled Taylor tail and its z derivative by the following.
        require(4*12*rho < Q(1, 2), "Geometric bound on the series tail")
        remainder = 4*layer_norm[k]*(d+1)**2 * 12**d * rho
        other = sum(zeta**p*layer_norm[p] for p in (1, 2) if p > k)
        other += shift*sum(map(abs, shift_modes))
        remainder += 32*other/(zeta**k*rho**(d-1))
        require(remainder < Q(1, 1024), "Scaled C1 error for all rows in the layer")
        radius = Q(1, 64)
        others = (1-radius)**(d-2)
        require(radius*others > remainder and others*(1-(d-2)*radius/(1-radius)) > remainder,
                "Signs and nonvanishing derivative near every simple integer root")
    require(tuple(map(len, expanded(n, vertical, modes, shift, Q(1)))) == (13, 9),
            "Exact support at the fixed sample shape for every nonzero mu")
    print("n=9, N=5: 80 simple moment zeros at fixed rational shaping parameters.")


def check_budgets():
    for m in range(7, 10000):
        n, r, vertical, total = budget(m)
        require(n == 3**r and 2*n+vertical-1 == m and m+1 < 8*n <= 3*(m+1),
                "Exact arbitrary budget")
        require(32*n*vertical >= 3*(m+1)**2 and total == vertical*sum(
            2*n//3**k * (2*3**(k-1)-1) for k in range(1, r+1)), "Uniform product and count")
    r = 37
    n, vertical = 3**r, 2*3**r
    m, total = 2*n+vertical-1, cycle_count(n, r, vertical)
    require(budget(m) == (n, r, vertical, total) and total > 6*m*m, "Same-budget strict gain")
    print("At m =", m, "the existence bound is", total, "and exceeds 6*m^2 by", total-6*m*m)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, metavar="M", help="print the exact allocation and cycle bound")
    args = parser.parse_args()
    if args.budget is not None:
        n, r, vertical, cycles = budget(args.budget)
        print("n, r, N =", n, r, vertical, "supports", (vertical+n-1, n), "cycles >=", cycles)
        return
    check_layers()
    check_rank_and_unfolding()
    check_support()
    check_two_layers_at_fixed_shape()
    check_budgets()
    print("Visibility, kernel determinants, row interpolation and separate real expansions passed.")


if __name__ == "__main__":
    main()
