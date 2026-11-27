"""Finite checks for the beta coefficients and separated scales at an odd turn."""
from fractions import Fraction
from itertools import pairwise

from flint import arb, ctx


def beta(x, y):
    return x.gamma() * y.gamma() / (x + y).gamma()


def stationary_jets(d):
    r = 2*d - 1
    b = d - 2
    half = Fraction(1, 2)
    quarter_r = Fraction(r, 4)
    # Ratios within each analytic row come from shifting the first beta argument.
    row0, row1 = [Fraction(1)], [Fraction(1)]
    for j in range(d):
        row0.append(row0[-1] * (j + half + quarter_r) / (j + 2 + quarter_r))
        row1.append(row1[-1] * (j + half - quarter_r) / (j + 1 - quarter_r))
    R0, R1 = row0[d] / row0[b], row1[d] / row1[b]
    assert 0 < R1 < R0
    assert row1[b+1] / row1[b] < row0[b+1] / row0[b]
    assert row1[d] / row1[b+1] == row0[d] / row0[b+1]

    # Use the two incomplete-integral constants in their beta form, independently
    # of the trigonometric simplification in the proof.
    three_halves = arb(3) / 2
    for j in range(b+1):
        alpha = arb(2*j+1) / r
        nu = alpha + arb(1) / 2
        c_plus = (beta(alpha, three_halves) + beta(alpha, -nu)) / r
        c_minus = beta(-nu, three_halves) / r
        assert c_plus*c_minus < 0, (d, j)

    # V'=(r+2)y^(2d)-r*y^(2d-2) annihilates the integral at both endpoints.
    # Check both analytic rows and the next fractional coefficient separately.
    for row in (row0, row1):
        assert (r+2)*row[d] == r*row[d-1]
    exceptional_alpha = arb(2*(d-1)+1) / r
    assert exceptional_alpha == 1
    assert exceptional_alpha.cos_pi() == -1 and exceptional_alpha.sin_pi() == 0
    alpha = arb(2*d+1) / r
    nu = alpha + arb(1)/2
    assert (beta(alpha, three_halves) + beta(alpha, -nu))/r < 0
    assert beta(-nu, three_halves)/r < 0
    assert r+4*d+2 == 3*r+4

    orders = sorted([0, 2*r, *(r + 4*j + 2 for j in range(b))])
    assert len(set(orders)) == d and orders[-1] < 3*r - 4
    for left, right in pairwise(orders):
        assert all((k-left)*(k-right) > 0 for k in orders if k not in (left,right))
    return R0, R1


if __name__ == "__main__":
    ctx.prec = 160
    degrees = (4, 5, 6, 9, 15, 31, 127, 129)
    for degree in degrees:
        upper, lower = stationary_jets(degree)
        if degree == 4:
            print(f"d=4: exact normalized ratios S1d/S1b={lower}, S0d/S0b={upper}.")
    print("Exact control determinants and adjacent-scale exponents passed for d =", *degrees)
    print(f"Both beta constants give opposite strict signs in {sum(d-1 for d in degrees)} cases (160-bit Arb).")
    print("S3.2: at j=d-1 both specified coefficients vanish. The analytic rows satisfy "
          "r*I_(d-1)=(r+2)*I_d coefficientwise through order one, and both fractional "
          "coefficients of I_d at exponent 3r+4 are strictly negative at all eight d.")
