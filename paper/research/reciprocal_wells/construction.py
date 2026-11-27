"""Rational primitives, controls and collected fields for the reciprocal-well bound."""

from dataclasses import dataclass
from fractions import Fraction as Q

from research.chebyshev_rows.verify import chebyshev
from research.exponent_variations.verify import inverse_and_determinant, matvec
from research.filtered_outlets.verify import at, divide_primitive, jet, powers, trim
from research.flat_minima.verify import changes, density
from research.mixed_outlet.verify import polynomial_product
from research.translated_wells.construction import Layer, affine, expanded_field, fit_primitive
from research.verify_term_bounds import require


def layers(n, L):
    if n < 2 or n & (n-1) or L < 2:
        raise ValueError("Require dyadic n >= 2 and L >= 2.")
    S = n*(3*L-1)//2
    result = []
    g = n
    while g >= 4:
        rows = tuple((n//g)*b for b in range(1, g, 2))
        frequencies = tuple(range(g//2, 2*S, g))
        signs = tuple((-1)**j for j in range(len(frequencies)))
        result.append(Layer(rows, frequencies, signs))
        g //= 2
    result.append(Layer((n//2,), tuple(range(1, 2*S, 2)), tuple((-1)**j for j in range(S))))
    return result


def model_count(a, M):
    if not 1 <= a < M:
        raise ValueError("The block requires M >= a+1.")
    return changes(a, M-a) + 2*(a-1)


def cycle_count(n, L, a, k):
    if n < 2 or n & (n-1) or not 1 <= a <= L//2 or k < 1:
        raise ValueError("Require dyadic n >= 2, L >= 2a >= 2 and k >= 1.")
    S = n*(3*L-1)//2
    return 4*k*sum(y*model_count(a, S//y) for y in (1 << i for i in range(n.bit_length()-1)))


def best_parameters(m):
    best, params = 0, None
    n = 2
    while 2*n <= m-2:
        for L in range(2, (m-2)//n+1):
            k = m-1-n*L
            for a in range(1, L//2+1):
                count = cycle_count(n, L, a, k)
                if count > best:
                    best, params = count, (n, L, a, k)
        n *= 2
    return best, params


def compose(outer, inner):
    result = [Q(0)]
    for coefficient in reversed(outer):
        result = trim(polynomial_product(result, inner))
        result[0] += coefficient
    return result


def baseline(n, L, a, t, kappa):
    """Return V(z), F(z), A'(F(z)) in ordinary ascending powers."""
    if t <= 0 or kappa <= 0 or not 1 <= a <= L//2:
        raise ValueError("Require t,kappa > 0 and L >= 2a >= 2.")
    C = [Q(1), Q(0), Q(-1)]
    q = L-2*a
    factor = [Q(1)] + [Q(0)]*q
    factor[q] += t**2
    outer = polynomial_product(powers(C, a), factor)
    for j, value in enumerate(C):
        outer[j] += kappa*t**(2*a-2)*value
    F = chebyshev(n)
    derivative = [j*outer[j] for j in range(1, len(outer))]
    return compose(outer, F), F, compose(derivative, F)


def high_model(a, M, q):
    """Coefficients of Z^(2j+1) reproducing the pinned nonnegative-exponent moments."""
    if not 1 <= a < M or not 0 < q < 1:
        raise ValueError("Require M >= a+1 >= 2 and 0 < q < 1.")
    old = density(a, M-a, q)
    return [Q(0)]*(a-1) + [Q(a, 2*j+1)*value for j, value in enumerate(old)]


def model_target(n, L, a, t, zeta, models):
    """Interpolate supplied odd numerators, already including any constant-density shift."""
    stages = layers(n, L)
    if len(models) != len(stages) or t <= 0 or not 0 < zeta < 1:
        raise ValueError("Supply one model per block, t > 0 and 0 < zeta < 1.")
    result = [Q(0)]*(n*(3*L-1))
    for ell, (stage, model) in enumerate(zip(stages, models)):
        size = len(stage.frequencies)
        if len(model) > size:
            raise ValueError("Odd model degree exceeds its block.")
        columns = [[value/(2*j+1) for j, value in enumerate(jet(Q(h, 2), n, size))]
                   for h in stage.frequencies]
        matrix = list(zip(*columns))
        rhs = [at(model, j)/t**(2*j) for j in range(size)]
        weights = matvec(inverse_and_determinant(matrix)[0], rhs)
        require(matvec(matrix, weights) == rhs, "Odd primitive jets reproduce the requested model")
        for h, sign, weight in zip(stage.frequencies, stage.signs, weights):
            factor = Q(n, h)*sign*weight*t**(2*a-2)*zeta**ell
            for j, coefficient in enumerate(chebyshev(h)):
                result[j] += factor*coefficient
    return trim(result)


@dataclass(frozen=True)
class ReciprocalControls:
    potential: list
    primitive: list
    denominator: list
    potential_change: list
    primitive_change: list
    exponent_change: list


def controls(n, L, a, t, kappa, eta, polynomial, logs):
    """Realize the numerator polynomial + logs(X)*log(X) in the translated chart."""
    p = n*L
    if not 0 < eta < 1 or len(polynomial) > 2*p-n+1 or len(logs) != p+1 or any(logs[:2]):
        raise ValueError("Invalid numerator degree, log columns or translation.")
    v, f, den = [affine(part, -1/eta, 1/eta) for part in baseline(n, L, a, t, kappa)]
    f += [Q(0)]*(p+1-len(f))
    minus_B, minus_H = divide_primitive(polynomial, den)
    B, H = [-x for x in minus_B], [-x for x in minus_H]
    require(len(B) <= p+1 and len(H) <= p+1, "Polynomial controls fit the force nodes")
    residual, theta = list(logs), [Q(0)]*(p+1)
    for j in range(2, p+1):
        pivot = f[j]*den[0]-v[j]
        if not pivot:
            raise ValueError("Singular exponent pivot at the supplied translation.")
        theta[j] = residual[j]/pivot
        residual[j] += theta[j]*v[j]
        if f[j]:
            for i, coefficient in enumerate(den):
                residual[j+i] -= theta[j]*f[j]*coefficient
    require(not any(residual), "All logarithmic numerator coefficients reproduced")
    return ReciprocalControls(v, f, den, H, B, theta)


def field_from_models(n, L, a, k, models, kappa, delta, t, zeta, eta, lam=None, mu=None):
    cycle_count(n, L, a, k)
    target = model_target(n, L, a, t, zeta, models)
    poly, logs = fit_primitive(target, n*L, n, eta)
    shape = controls(n, L, a, t, kappa, eta, poly, logs)
    return expanded_field(shape, a, k, delta, t, lam, mu)
