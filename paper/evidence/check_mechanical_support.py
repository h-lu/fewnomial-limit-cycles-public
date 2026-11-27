"""Exact finite checks for the mechanical-renewal comparison in Section S1.2."""
from fractions import Fraction
from math import comb, isqrt

from sympy import Matrix, Poly, Rational, diff, symbols


def source_counts(n):
    """Count the allowed birth modes directly and compare with Proposition 3.3."""
    stages = n.bit_length() - 1
    selected = 0
    for i in range(1, stages + 1):
        size = n >> (stages - i)
        previous, extra = divmod(size, 2)
        births = 0
        remainder_loss = 0
        for k in range(i):
            copies = 2**k
            half_degree, remainder = divmod(previous, copies)
            dimension = sum(size - copies*(2*q + 1) for q in range(half_degree))
            assert dimension == half_degree * (size - copies*half_degree)
            assert 4*copies*dimension + (size-2*copies*half_degree)**2 == size**2
            births += 4*copies*(dimension - 1)
            remainder_loss += remainder**2 + extra*remainder
        assert births == i*(size**2-extra) - 4*remainder_loss - 4*(2**i-1)
        assert births <= i*size**2
        selected = 4*selected + births
    assert 6*selected >= n**2*(3*stages**2 + stages - 44)
    if n == 2**stages:
        assert 6*selected == 3*n**2*stages*(stages+1) - 16*n**2 + 24*n - 8
    return selected


def budgets(n):
    degree = 2*n - 1
    slots = sum(degree - odd + 1 for odd in range(1, degree+1, 2))
    assert slots == n**2
    first, last = 2*n**2, 2*(n+1)**2 - 1
    # The whole exact-N window lies in the ordinary degree space. These
    # endpoint inequalities also give 2r+1 <= log_2(N) < 2r+3.
    r = n.bit_length() - 1
    assert last <= (degree+1)*(degree+2)
    assert last-first == 4*n+1
    assert 2**(2*r+1) <= first <= last < 2**(2*r+3)
    for terms in (first, last):
        assert isqrt(terms//2) == n
    return first, last, degree


def check_global_form():
    """Expand a non-symmetric seed. The two parities apply after the fold."""
    x, y = symbols("x y")
    a, b = Rational(3, 2), Rational(5, 3)
    u, v = x**2-a, y**2-b
    seed_dx = 1 + 2*u - 3*v + 5*u*v**2
    seed_dy = 3 - u + v**2 - 7*u**2*v
    hamiltonian = (u**4 + x**10) - (v**4 + 2*y**10)
    vertical = 2*y*(x*(1+v+v**2) + 5*x**3)
    horizontal = 2*x*(3*y*(1-u+u**2) - y**3)
    first = Poly(diff(hamiltonian, y) - Rational(2, 7)*(2*y*seed_dy + vertical), x, y)
    second = Poly(-diff(hamiltonian, x) + Rational(2, 7)*(2*x*seed_dx + horizontal), x, y)
    assert all(j % 2 for i, j in first.monoms())
    assert all(i % 2 for i, j in second.monoms())
    assert max(first.total_degree(), second.total_degree()) == 9
    print("Rational two-coordinate fold, even Hamiltonian and both axis sources: physical component parities verified.")


def check_last_birth(n):
    """Read the chosen physical coefficient rows from translated birth polynomials."""
    Y = symbols("Y")
    b = Rational(5, 3)
    modes = 0
    for p in range(n-1):
        width = (n-2-p)//2 + 1
        columns = [Poly((Y-b)**(2*q), Y) for q in range(width)]
        block = Matrix(width, width, lambda j, q: columns[q].nth(2*j))
        for j in range(width):
            for q in range(width):
                expected = comb(2*q, 2*j)*(-b)**(2*(q-j)) if q >= j else 0
                assert block[j,q] == expected
        assert block.det() == 1
        modes += width
    assert modes == n*n//4
    print(f"n={n}: {modes} independent translated birth rows per axis, {2*modes} generic mixed component positions.")


def check_sparse_five():
    # Half-degrees 1,2,5. The global form at half-degree two may be zero.
    # The independent future-label forms receive the two auxiliary births.
    def modes(previous, copies):
        return [(p, q) for q in range(previous)
                for p in range(5 - copies*(2*q+1))]
    initial = modes(1, 2)
    fresh = modes(2, 1)
    assert (len(initial), len(fresh)) == (3, 6)
    per_axis_branch = 2*(len(initial)-1) + len(fresh)-1
    assert source_counts(2) == 0 and 4*per_axis_branch == source_counts(5) == 36
    # Auxiliary pullback and new birth at level 1, then the final axis transplant.
    x, y, w = symbols("x y w")
    shift = Rational(3, 2)
    aux = {}
    for p, q in initial:
        aux[p] = aux.get(p, 0) + 2*w*(w**2-shift)**(2*q)
    for p, q in fresh:
        aux[p] = aux.get(p, 0) + w**(2*q)
    axis = Poly(sum(2*y*x**(2*p+1)*rp.subs(w, y**2-shift) for p, rp in aux.items()), x, y)
    mixed = {(2*i+1, 2*j+1) for i in range(4) for j in range(4-i)}
    assert set(axis.monoms()) <= mixed and len(mixed) == 10 and axis.total_degree() <= 8
    slots = 2*(5+len(mixed))
    assert slots == 30 and slots <= 33 <= 32*33
    print("n=5, half-degrees 1,2,5: zero first global form, 9 auxiliary zeros per branch, 36 global.")
    print("Per component at most 5 pure and 10 mixed slots, degree <= 9. Filling to 33 at cap 31 is feasible.")


if __name__ == "__main__":
    check_global_form()
    check_sparse_five()
    for n in range(2, 257):
        count = source_counts(n)
        first, last, degree = budgets(n)
        r = n.bit_length() - 1
        assert r*(r+1)//2 <= 4**(r-1)
        assert 4*count <= 2*n**2*r*(r+1) <= n**4 < 4*((first-2)**2//4)
        if n in (4, 8, 16, 32):
            print(f"n={n}: exact N from {first} through {last}, degree cap {degree}, integer recurrence value {count}.")
    print("All n=2..256: parity slots, floor windows, full capacity, birth recurrence and finite lower inequality passed.")
    for n in (2, 3, 7, 8, 17, 24):
        check_last_birth(n)
    coefficient = Fraction(1, 2)*Fraction(1, 2)*Fraction(1, 2)**2
    assert coefficient == Fraction(1, 16)
    print(f"Parity-budget substitution gives coefficient {coefficient} in the integer recurrence. Section S1.2 proves the term transfer from the common hyperbolic realization of Lu, Proposition 3.3.")
