"""Exact support examples and parameter counts for mixed and reciprocal polynomial wells."""

from fractions import Fraction as Q
from math import comb, factorial

from research.chebyshev_rows.verify import chebyshev
from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.filtered_outlets.verify import at, jet, trim
from research.mixed_outlet.verify import polynomial_product
from research.reciprocal_wells.construction import controls, high_model, model_count
from research.translated_wells.construction import affine, expanded_field
from research.verify_term_bounds import require


def blocks(n, limit):
    if n < 2 or n & (n-1):
        raise ValueError("The well index n must be a power of two of at least two.")
    return [(b, range(b, limit+1, 2*b)) for b in (n >> k for k in range(1, n.bit_length()))]


def linear_count(n, d, wells):
    if d < 2*n or wells < 1:
        raise ValueError("Require d >= 2n and a positive horizontal well count.")
    return wells*sum(2*b*(len(freq)-1) for b, freq in blocks(n, d))


def quotient_degree(n, L):
    p = n*L
    return 3*p-n-1+(p-1)//(n-1)


def reciprocal_count(n, L, a, k):
    if not 1 <= a <= L//2 or k < 1:
        raise ValueError("Require L >= 2a >= 2 and k >= 1.")
    return 4*k*sum(b*model_count(a, len(freq)) for b, freq in blocks(n, quotient_degree(n, L)))


def best_reciprocal(m):
    best, params = 0, None
    for exponent in range(1, m.bit_length()):
        n = 2**exponent
        for L in range(2, (m-2)//n+1):
            k = m-1-n*L
            for a in range(1, L//2+1):
                value = reciprocal_count(n, L, a, k)
                if value > best:
                    best, params = value, (n, L, a, k)
    return best, params


def chebyshev_target(n, limit, models, t, zeta, multiplier=1):
    """Fit the odd primitive on each dyadic block. Models list odd coefficients."""
    stages = blocks(n, limit)
    if len(models) != len(stages) or t <= 0 or zeta <= 0:
        raise ValueError("Supply one model per block and positive scales.")
    result = [Q(0)]*(multiplier*limit+1)
    for level, ((_, freq), model) in enumerate(zip(stages, models)):
        size = len(freq)
        if not 0 < len(model) <= size:
            raise ValueError("Odd target degree outside this block.")
        columns = [[value/(2*j+1) for j, value in enumerate(jet(Q(h, 2), n, size))] for h in freq]
        rows = list(zip(*columns))
        rhs = [at(model, j)/t**(2*j) for j in range(size)]
        weights = matvec(inverse_and_determinant(rows)[0], rhs)
        require(matvec(rows, weights) == rhs, "Recovered the prescribed odd coefficients")
        for j, (h, weight) in enumerate(zip(freq, weights)):
            factor = Q(n, h)*(-1)**j*weight*zeta**level
            for power, value in enumerate(chebyshev(multiplier*h)):
                result[power] += factor*value
    return trim(result)


def log_column(j, b):
    if b > j:
        return Q((-1)**(b-j-1)*factorial(j)*factorial(b-j-1), factorial(b))
    return sum((Q((-1)**(b-v-1)*comb(j, v), b-v) for v in range(b)), Q(0))


def quotient_fit(n, L, target, eta):
    """Fit in ordinary X powers, logarithmic X powers and globally invisible Chebyshev modes."""
    p, K, top = n*L, 2*n*L-n, quotient_degree(n, L)
    if not 0 < eta < 1 or len(target) > top+1:
        raise ValueError("Translation or target is outside the interpolation range.")
    high = {h: chebyshev(h) for h in range(K+n, top+1, n)}
    rows, rhs = [], []
    for b in range(K+1, top+1):
        rows.append([log_column(j, b) for j in range(2, p+1)]
                    + [at(poly, b)*eta**(h-b) for h, poly in high.items()])
        rhs.append(at(target, b)/eta**b)
    weights = matvec(inverse_and_determinant(rows)[0], rhs)
    logs = [Q(0), Q(0)] + weights[:p-1]
    blind = {h: value*eta**h for h, value in zip(high, weights[p-1:])}
    low = []
    for b in range(K+1):
        remainder = at(target, b)-sum(blind[h]*at(poly, b) for h, poly in high.items())
        low.append(remainder/eta**b-sum(logs[j]*log_column(j, b) for j in range(2, p+1)))
    return affine(low, -1, 1), logs, blind


def reciprocal_sample():
    n, L, a, k = 4, 2, 1, 3
    t, zeta, eta, kappa, delta = Q(1, 8), Q(1, 8), Q(1, 16), Q(1, 4), Q(1, 16)
    top = quotient_degree(n, L)
    models = []
    for level, (_, freq) in enumerate(blocks(n, top)):
        model = high_model(a, len(freq), Q(1, 4))
        model[0] += Q((-1)**level, 17)*(a+kappa)
        models.append(model)
    target = chebyshev_target(n, top, models, t, zeta)
    poly, logs, blind = quotient_fit(n, L, target, eta)
    shape = controls(n, L, a, t, kappa, eta, poly, logs)
    return expanded_field(shape, a, k, delta, t), (n, L, eta, target, poly, logs, blind)


def half_gamma(j):
    return Q(factorial(2*j), 4**j*factorial(j))


def disk_model(size, product):
    degree = size-1
    if product:
        density = [Q((-1)**j*comb(degree, j), half_gamma(j)) for j in range(size)]
    else:
        poly = [Q(1)]
        for root in range(1, size):
            poly = polynomial_product(poly, [-root, 1])
        density = [value*factorial(j+1)/half_gamma(j) for j, value in enumerate(poly)]
    return [value/(2*j+1) for j, value in enumerate(density)]


def derivative(poly):
    return [j*value for j, value in enumerate(poly) if j]


def shaped_well(n, d, product, t=Q(1, 16), zeta=Q(1, 7), filling=Q(1, 10001)):
    shape = chebyshev_target(n, d, [disk_model(len(freq), product) for _, freq in blocks(n, d)],
                             t, zeta, multiplier=2)
    shape += [Q(0)]*(2*d+1-len(shape))
    for j in range(1, d+1):
        shape[2*j] += filling
    J = chebyshev(2*n)
    V = [-v/(2*t*t) for v in polynomial_product(J, J)]
    V[0] += 1/(2*t*t)
    return V, J, shape


def assemble(U, V, Bx, By, S, R, lam, eta, mu):
    """Equation for the affine family, with the irrelevant constant removed from By."""
    def add(part, index, value):
        part[index] = part.get(index, 0)+value

    P, Qpart = {}, {}
    for poly, scale in ((V, 1), (R, eta)):
        for y, c in enumerate(derivative(poly)):
            add(P, (0, y), scale*c)
    for poly, scale in ((U, -1), (S, -lam)):
        for x, c in enumerate(derivative(poly)):
            add(Qpart, (x, 0), scale*c)
    for x, cx in enumerate(derivative(Bx)):
        for y, cy in enumerate(By[1:], 1):
            if cx and cy:
                add(Qpart, (x, y), mu*cx*cy)
    return tuple({key: value for key, value in part.items() if value} for part in (P, Qpart))


def mixed_sample(kind):
    if kind == "six":
        tau = Q(1, 2**40)
        V = [Q(1, 2), 0, -1, 0, Q(1, 2)]
        R = [Q(-5, 8), 0, Q(21, 8), 0, -3, 0, 1]
        return assemble([0, 0, Q(1, 2)], V, [0, 1], V, [0], R, 0, tau, tau)
    tau = Q(1, 4096)
    V, By, R = shaped_well(4, 8, kind != "linear")
    if kind == "linear":
        kinetic = chebyshev(5)
        U = [128*c for c in polynomial_product(kinetic, kinetic)]
        return assemble(U, V, [0, 1], By, [0], R, 0, tau, tau)
    if kind == "grid":
        U, Bx, S = shaped_well(2, 6, True)
        return assemble(U, V, Bx, By, S, R, tau, tau, tau)
    if kind == "packets":
        exponents = (1, 25, 81)
        S = [Q(0)]*(2*exponents[-1]+2)
        for e in exponents:
            S[2*e+1] = 1/(half_gamma(e)*(2*e+1))
        U = [0, 0, Q(1, 2)]
        return assemble(U, V, U, By, S, R, tau, tau, tau)
    raise ValueError("Unknown mixed field sample.")
