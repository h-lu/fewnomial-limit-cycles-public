"""Rational construction in the coordinates of bound.tex. Coefficients ascend."""

from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb, factorial, lcm

from research.chebyshev_rows.verify import chebyshev
from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.filtered_outlets.verify import (
    at, blocks, collect, divide_primitive, even_well, jet, powers, trim,
)
from research.flat_minima.verify import changes, density
from research.verify_term_bounds import require


def affine(poly, offset, slope):
    """Expand poly(offset + slope*x) in ordinary powers of x."""
    result = [Q(0)]
    for value in reversed(poly):
        result = [offset*result[0]+value] + [
            slope*left + offset*right for left, right in zip(result, result[1:])
        ] + [slope*result[-1]]
    return trim(result)


@dataclass(frozen=True)
class Layer:
    rows: tuple
    frequencies: tuple
    signs: tuple


def layers(n, S):
    if n < 2 or n & (n-1) or S < n or S % n:
        raise ValueError("The grid size must be dyadic and divide S.")
    result = [
        Layer(tuple(i//2 for i in block.rows), tuple(2*s for s in block.modes), block.signs)
        for block in blocks("integer", n, S)
    ]
    odd = tuple(range(1, 2*S, 2))
    result.append(Layer((n//2,), odd, tuple((-1)**(h//2) for h in odd)))
    return result


def cycle_count(n, L, a, k):
    if n < 2 or n & (n-1) or not 1 <= a < L or k < 1:
        raise ValueError("Require dyadic n >= 2, L > a >= 1 and k >= 1.")
    S = n*(3*L-a)
    total = changes(a, S-1)
    g = n
    while g >= 4:
        total += (g//2)*changes(a, 2*S//g-1)
        g //= 2
    return 4*k*total


def model_target(n, L, a, q_values, shifts, t, zeta):
    """Build T(z) by the separate even and odd frequency solves."""
    stages = layers(n, n*(3*L-a))
    if len(q_values) != len(stages) or len(shifts) != len(stages):
        raise ValueError("Supply one q and one constant shift per block.")
    if t <= 0 or not 0 < zeta < 1 or not all(0 < q < 1 for q in q_values):
        raise ValueError("Require t > 0 and the block weights q and zeta in (0,1).")
    primitive = [Q(0)] * (2*n*(3*L-a))
    for ell, (stage, q, shift) in enumerate(zip(stages, q_values, shifts), 1):
        M = len(stage.frequencies)
        model = density(a, M-1, q)
        model[0] += shift
        matrix = list(zip(*(jet(Q(h, 2), n, M) for h in stage.frequencies)))
        rhs = [value / t**(2*j) for j, value in enumerate(model)]
        weights = matvec(inverse_and_determinant(matrix)[0], rhs)
        require(matvec(matrix, weights) == rhs, "Every block reproduces its model jets")
        for h, sign, weight in zip(stage.frequencies, stage.signs, weights):
            factor = zeta**ell * sign * weight / h
            for j, value in enumerate(chebyshev(h)):
                primitive[j] += factor*value
    primitive[0] = Q(0)
    return trim(primitive)


def differences(values):
    """Leading forward difference in each order at the first input node."""
    result = []
    while values:
        result.append(values[0])
        values = [right-left for left, right in zip(values, values[1:])]
    return result


def fit_primitive(target, p, b, eta):
    """Fit the Taylor coefficients of P(X) + log(X) sum l_j X^j.

    For h > K, multiply each row by (-1)^(h-1) h!/(h-p-1)!.
    Column j then equals (-1)^j j! times the rising factorial of
    h-p of order p-j. Interpolation in h recovers all log coefficients.
    """
    K = 2*p-b
    D = K+p-1
    if not (0 < b < p and eta > 0 and len(target) <= D+1):
        raise ValueError("Target degree or translation is outside Lemma 3.")
    jets = [at(target, h)/eta**h for h in range(D+1)]
    normalized = [(-1)**(h-1)*jets[h]*Q(factorial(h), factorial(h-p-1))
                  for h in range(K+1, D+1)]
    newton = differences(normalized)
    # Evaluate the Newton polynomial at h=p,p-1,... . Here h-(K+1)<0.
    backward = [
        sum(value*(-1)**d*comb(K+r-p+d, d) for d, value in enumerate(newton))
        for r in range(p-1)
    ]
    logs = [Q(0)] * (p+1)
    for d, value in enumerate(differences(backward)):
        logs[p-d] = (-1)**p * value / (factorial(d)*factorial(p-d))

    # Expand the log multiplier once and use the series of log(1+w).
    weight = affine(logs, 1, 1)
    log_series = [Q(0)] + [Q((-1)**(h-1), h) for h in range(1, D+1)]
    remainder = [
        goal - sum(weight[j]*log_series[h-j] for j in range(min(h, len(weight)-1)+1))
        for h, goal in enumerate(jets)
    ]
    require(not any(remainder[K+1:]), "Highest translated Taylor rows vanish exactly")
    return affine(remainder[:K+1], -1, 1), logs


@dataclass(frozen=True)
class Controls:
    potential: list
    primitive: list
    quotient: list
    potential_change: list
    primitive_change: list
    exponent_change: list


def controls(n, L, a, eta, polynomial, logs):
    """Read the actual X coefficients and solve the triangular exponent equations."""
    p = 2*n*L
    c_even = even_well("integer", n)
    c_z = [at(c_even, j//2) if j % 2 == 0 else 0 for j in range(2*n+1)]
    c = affine(c_z, -1/eta, 1/eta)
    v, f = powers(c, a), powers(c, L)
    u = [Q(L, a)*value for value in powers(c, L-a)]
    A, B = divide_primitive(polynomial, u)
    require(len(A) <= p+1 and len(B) <= p+1, "Polynomial shape controls fit the force")
    residual = list(logs)
    theta = [Q(0)]*(p+1)
    for j in range(2, p+1):
        vj = at(v, j)
        pivot = f[j]-vj*u[0]
        if not pivot:
            raise ValueError("Singular pivot at supplied eta. Choose a smaller translation.")
        theta[j] = residual[j]/pivot
        residual[j] -= theta[j]*f[j]
        if vj:
            for i, value in enumerate(u):
                residual[j+i] += theta[j]*vj*value
    require(not any(residual), "Exponent changes reproduce the entire logarithmic polynomial")
    return Controls(v, f, u, A, B, theta)


def expanded_field(shapes, a, k, delta, t, lam=None, mu=None):
    """Expand the two components. Automatic lambda and mu enforce support conditions."""
    p = len(shapes.exponent_change)-1
    if not (a >= 1 and k >= 1 and t > 0 and 0 < delta < 1):
        raise ValueError("Require positive orders and t, and 0 < delta < 1.")
    slopes = shapes.exponent_change
    if lam is None:
        lam = Q(1, 8)/(1+max(map(abs, slopes)))
        while any(at(shapes.potential, j)+lam*at(shapes.potential_change, j)+lam**2 == 0
                  for j in range(1, p+1)):
            lam /= 2
    if lam <= 0 or max(map(abs, slopes))*lam >= Q(1, 4):
        raise ValueError("Lambda must preserve the separated exponent nodes.")
    nodes, g, h = [], [], []
    for j in range(1, p+1):
        node = j+lam*slopes[j]
        amplitude = at(shapes.potential, j)+lam*at(shapes.potential_change, j)+lam**2
        nodes.append(node)
        g.append(-t**(-2*a)*node*amplitude)
        h.append(node*(shapes.primitive[j]+lam*at(shapes.primitive_change, j)))
    if not all(g):
        raise ValueError("Lambda cancels a restoring monomial.")
    ratios = [v/w for v, w in zip(h, g)]
    if mu is None:
        mu = Q(1, 8)/(1+max(map(abs, ratios)))
    if mu <= 0 or mu*max(map(abs, ratios)) >= Q(1, 2):
        raise ValueError("Mu must preserve positive lifted exponents.")
    tilts = [mu*value for value in ratios]
    rates = [delta**i for i in range(1, k+1)]
    D = 2*lcm(*(v.denominator for v in nodes+rates+tilts))

    def scaled(v):
        result = D*v
        require(result.denominator == 1 and result >= 0 and result % 2 == 0,
                "Lifted power is an even nonnegative integer")
        return int(result)

    P = collect([(1, 2*D, sum(rates))] + [(1, scaled(1+v), -v) for v in rates])
    Qpart = collect([(scaled(node), 1+scaled(1+tilt), coeff)
                     for node, tilt, coeff in zip(nodes, tilts, g)])
    require((len(P), len(Qpart)) == (k+1, p), "Separate collected real supports")
    # Recover the log field from ordinary integer exponents, including its perturbation.
    recovered_D = max(y for _, y in P)//2
    require({(Q(y, recovered_D)-1, coeff) for (_, y), coeff in P.items()}
            == {(Q(1), sum(rates))} | {(v, -v) for v in rates}, "Shelf coefficient readback")
    for j, ((x, y), coeff) in enumerate(sorted(Qpart.items())):
        require(Q(x, recovered_D) == nodes[j] and coeff == g[j]
                and coeff*(Q(y-1, recovered_D)-1) == mu*h[j], "Restoring term readback")
    return P, Qpart
