"""Outward rational enclosures for the new low powers and a positive regularization."""

from fractions import Fraction as Q
from functools import cache

from research.flat_minima.verify import binomial, endpoint_index
from research.reciprocal_wells.construction import high_model, model_count
from research.separated_exponents.verify import interval_add_scaled
from research.verify_term_bounds import require


@cache
def kernel_interval(a, j, side, root, last):
    """Enclose I_gamma at h=root^(2a) or 1-root^(2a), with 0 < root < 1."""
    degree = 2*j+3-2*a
    gamma = Q(degree, 2*a)
    v = root**(2*a)
    if side == "lower":
        lead = root**(degree+2*a)/(gamma+1)
        return lead, lead/(1-v)
    if side != "upper" or last <= gamma:
        raise ValueError("Require a known endpoint and a convergent tail majorant.")
    term = lambda ell: binomial(gamma, ell)/(gamma-ell)
    # Horner evaluation keeps exact denominators small for the extreme marks.
    regular = term(last)
    for ell in reversed(range(last)):
        regular = term(ell) - v*regular
    tail = abs(term(last+1))*v**(last+1)/(1-v)
    floor = gamma.numerator//gamma.denominator
    theta = gamma-floor
    upper = 1/theta + 1/(1-theta)
    factor = (-1)**(floor+1)*root**degree
    return interval_add_scaled((regular-tail, regular+tail), (upper/2, upper), factor)


def sign_margin(interval, sign):
    return min(sign*v for v in interval)


def smaller_dyadic(bound):
    """A power of two strictly between zero and a positive rational bound."""
    if bound <= 0:
        raise ValueError("Positive bound required.")
    exponent = max(1, bound.denominator.bit_length()-bound.numerator.bit_length()+2)
    return Q(1, 1 << exponent)


def certified_model(a, M, q=Q(1, 256)):
    d, last = M-a, M+2
    coeff = high_model(a, M, q)
    limit = endpoint_index(a, d)[3]+3
    marks = []
    for i in range(d+1):
        _, fractional, low, exponent, _ = endpoint_index(a, i)
        if fractional:
            marks.append(("lower", q**(2*(limit-exponent)), (-1)**low))
    for i in reversed(range(d+1)):
        exponent = endpoint_index(a, i)[3]
        marks.append(("upper", q**(2*(limit-exponent)), (-1)**i))

    def enclosure(side, root):
        bounds = Q(0), Q(0)
        for j, c in enumerate(coeff):
            if c:
                bounds = interval_add_scaled(bounds, kernel_interval(a, j, side, root, last), c)
        return bounds

    def margins():
        return [sign_margin(enclosure(side, root), sign) for side, root, sign in marks]

    require(min(margins()) > 0 and marks[0][2] == marks[-1][2] == 1,
            "Original high-mode signs hold in the reciprocal integral")
    for j in reversed(range(a-1)):
        old_margins = margins()
        capacities = []
        for (side, root, _), margin in zip(marks, old_margins):
            bounds = kernel_interval(a, j, side, root, last)
            capacities.append(margin/(4*max(map(abs, bounds))))
        sign = -marks[0][2]
        coeff[j] = sign*smaller_dyadic(min(capacities))
        require(min(margins()) > 0, "Low power preserves all established strict signs")
        for side, pos in (("lower", 0), ("upper", -1)):
            root = marks[pos][1]**2
            while sign_margin(enclosure(side, root), sign) <= 0:
                root *= root
            if side == "lower":
                marks.insert(0, (side, root, sign))
            else:
                marks.append((side, root, sign))

    energies = [x**(2*a) if side == "lower" else 1-x**(2*a) for side, x, _ in marks]
    require(all(x < y for x, y in zip(energies, energies[1:])), "Strictly ordered marks")
    changes = sum(left[2] != right[2] for left, right in zip(marks, marks[1:]))
    require(changes == model_count(a, M), "Every additional pair of endpoint signs")
    margin = Q(2, a)*min(margins())
    min_v = 1-max(energies)
    norm = sum(map(abs, coeff))
    # For s=kappa^(1/(2a)), the root moves by at most s. The denominator
    # changes by at most kappa on [0,1]. These bounds apply to every mark.
    s = smaller_dyadic(min(Q(1), margin*min_v**2/(16*norm)))
    kappa = s**(2*a)
    error = 4*norm*(kappa/min_v**2+s/min_v)
    require(error < margin/2, "One positive rational kappa preserves all marked signs")
    return coeff, marks, kappa
