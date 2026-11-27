"""Exact checks for the two-axis obstruction and the fixed nine-term field.

Run python3 -B -m research.cross_visibility.verify from the repository root.
"""

import gzip
from argparse import ArgumentParser
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path

from research.chebyshev_rows.verify import chebyshev
from research.filtered_outlets.verify import blocks, comparators
from research.flat_minima.verify import changes
from research.mixed_outlet.verify import EPSILON_POWER, T, collect
from research.saddle_moments.verify import envelope
from research.separated_exponents.witness import fingerprint
from research.tail_interpolation.verify import expanded_field as eight_field
from research.tail_interpolation.witness import check_brackets, check_transfer
from research.verify_term_bounds import require


FIELD_PATH = Path(__file__).with_name("field.hex.gz")
DELTA = Q(1, 2**6144)
PRIME = 65521
EXPECTED_FIELD = "56371e8a3da6793d1426aca2dd18121a48e430566afe58d18dde4ef381f84c8e"


def derivative(n):
    c = chebyshev(n)
    return {i-1: i*c[i] for i in range(1, n+1) if c[i]}


def divergence(modes):
    return collect((i, j, coeff*a*b)
                   for k, l, coeff in modes for i, a in derivative(k).items()
                   for j, b in derivative(l).items())


def rank(rows):
    """Rectangular column rank in the fixed prime field."""
    pivots = {}
    for row in rows:
        v = [x % PRIME for x in row]
        for j in range(len(v)):
            if not v[j]:
                continue
            if j in pivots:
                factor = v[j]
                v[j:] = [(x-factor*y) % PRIME for x, y in zip(v[j:], pivots[j])]
            else:
                scale = pow(v[j], -1, PRIME)
                pivots[j] = [x*scale % PRIME for x in v[j:]]
                break
    return len(pivots)


def freq_series(k, n, length):
    alpha = k*pow(n, -1, PRIME) % PRIME
    a = (1-alpha)*pow(2, -1, PRIME) % PRIME
    b = (1+alpha)*pow(2, -1, PRIME) % PRIME
    out = [1]
    for i in range(1, length):
        out.append(out[-1]*(a+i-1)*(b+i-1)*pow(i, -1, PRIME) % PRIME)
    return out


def fourier_sine(index, n, generator):
    root = pow(generator, (PRIME-1)//(4*n), PRIME)
    return (pow(root, index, PRIME)-pow(root, -index, PRIME)) % PRIME


def check_at_aspect(n, N, klist, llist, aspect, generator, full_kernel_rank):
    pairs = [(k, l) for k in klist for l in llist]
    require(len({min(k % (2*n), -k % (2*n)) for k in klist}) == len(klist),
            "Horizontal signed residues distinct")
    length, xi = 48, pow(aspect, -1, PRIME)
    columns = []
    for k, l in pairs:
        x, y = freq_series(k, n, length), freq_series(l, N, length)
        y = [val*pow(xi, j, PRIME) % PRIME for j, val in enumerate(y)]
        columns.append([sum(x[j]*y[i-j] for j in range(i+1)) % PRIME for i in range(length)])
    rows = list(zip(*columns))
    kernel_rank = rank(rows)
    require((kernel_rank == len(pairs)) == full_kernel_rank, "Single-cell kernel rank")
    cross_rows = []
    for u in range(1, 2*n, 2):
        for v in range(1, 2*N, 2):
            if u == 3 or v == 3:
                w = [fourier_sine(k*u, n, generator)*fourier_sine(l*v, N, generator)
                     for k, l in pairs]
                cross_rows.extend([[x*y % PRIME for x, y in zip(row, w)] for row in rows])
    hidden = [(k, l) for k, l in pairs if l%(2*N) == 0 or
              (3*k%(2*n) == 0 and 3*l%(2*N) == 0)]
    require(rank(cross_rows) == len(pairs)-len(hidden), "Exact finite-jet cross nullity")
    print(f"At aspect {aspect} the {len(pairs)} kernels have rank {kernel_rank} mod {PRIME}. "
          f"The invisible cross has nullity {len(hidden)}.")


def check_cross_rank():
    require(all(PRIME % p for p in range(2, isqrt(PRIME)+1)), "Prime field")
    prime_divisors = [p for p in range(2, PRIME) if (PRIME-1)%p == 0
                      and all(p%d for d in range(2, isqrt(p)+1))]
    generator = next(g for g in range(2, PRIME)
                     if all(pow(g, (PRIME-1)//p, PRIME) != 1 for p in prime_divisors))
    check_at_aspect(9, 15, (2, 9, 29, 42), (3, 10, 20, 30, 41), 17, generator, True)
    check_at_aspect(9, 9, (2, 6, 9, 11), (2, 6, 9, 11, 18), 1, generator, False)


def check_structural_examples():
    for n, N in [(3, 3), (9, 15), (21, 9), (27, 27)]:
        ax, ay = 2*n//3, 2*N//3
        base = divergence([(ax, ay, 1)])
        require(len(base) == n*N//9, "Sharp divergence support")
        require(len(divergence([(n, N, 1)])) == ((n+1)//2)*((N+1)//2), "Jacobian product support")
        c = Q(1023, 1024)
        x1, x2 = chebyshev(ax), chebyshev(2*ax)
        vertical = derivative(ay)
        first = collect((i, j, (Q(x2[i], 2*ax)-(c*x1[i]/ax if i < len(x1) else 0))*z/ay)
                        for i in range(1, 2*ax+1) for j, z in vertical.items())
        require(len(first) == ax*ay//2 and all(i and i%2 == 0 and j%2 for i, j in first),
                "Collected support and disjoint restoring powers")
        require(-Q((2*ax)**2-ax**2, 8*n**2) == -Q(1, 6), "Simple normalized germ slope")
        require(sum(u%3 != 0 for u in range(1, 2*n, 2)) *
                sum(v%3 != 0 for v in range(1, 2*N, 2)) == 4*n*N//9, "Distinct exposed cells")
        if n == N == 3:
            require(len(first) == n*N//9 + min(n, N)//3 == 2, "First zero at the threshold")
    # Arbitrarily spaced powers on the three horizontal slots at d_x=6.
    # The nonzero Chebyshev critical points have squares 1/4 and 3/4.
    nodes = (Q(1, 4), Q(3, 4))
    evaluation = [[z**i for i in (0, 2, 5)] for z in nodes]
    a, b = evaluation
    kernel = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    require(all(kernel) and all(sum(x*y for x, y in zip(kernel, row)) == 0 for row in evaluation),
            "One-dimensional critical-point constraints on powers 1, 5, 11")
    require(kernel[0] > 0 > kernel[1] and kernel[2] > 0, "Both positive-root slots are occupied")
    base, silent = divergence([(6, 10, 1)]), divergence([(2, 30, 1)])
    weight = -Q(base[1, 1], silent[1, 1])
    combined = divergence([(6, 10, 1), (2, 30, weight)])
    require((1, 1) not in combined and len(combined) >= 15, "Cancellation by a globally blind harmonic")
    at_half = sum(c*Q(1, 2)**i for (i, j), c in combined.items() if j == 1)
    require(at_half and sum(c*Q(1, 2)**i for i, c in derivative(6).items()) == 0,
            "Horizontal derivative divisibility need not hold for the whole divergence")
    print("Sharp and two-mode supports checked; a blind harmonic cancels an actual ordinary monomial.")


def check_nine_estimates():
    check_brackets()
    check_transfer()
    root, rho = Q(1, 2**3072), Q(1, 2**1536)
    eta = 512*DELTA
    require(root**2 == rho**4 == DELTA and DELTA**4 == T**3, "Fixed scales and last coefficient")
    require(eta + Q(1, 2**128) < Q(1, 2**112) and 2*root < Q(1, 2**91),
            "Both finite turning-point margins")
    require(2**64*root < Q(1, 2**3000), "Horizontal moment value")
    require(2**105*root + 2**246*Q(1, 2**768) + 128*2**50*eta*2**2304 < Q(1, 2**520),
            "Horizontal energy derivative")
    require(2**230*T*T*2**200 < Q(1, 2**15000) and EPSILON_POWER == 1024 and
            8+8192*(429-EPSILON_POWER) < -1000, "Vertical and nonlinear transfer")
    require(Q(1, 2**3000)+Q(1, 2**15000)+Q(1, 2**1000) < Q(1, 2**710) and
            Q(1, 2**520)+Q(1, 2**15000)+Q(1, 2**1000) < Q(1, 2**200), "Full bracket signs and slopes")
    old_flat, old_rows, old_separated, gs = comparators(9)
    filtered = max(2*4*sum(len(b.rows)*changes(1, len(b.modes)-1) for b in blocks(grid, 4, 4))
                   for grid in ("half", "integer"))
    require((old_flat, old_rows, old_separated, gs, filtered, envelope(9)) ==
            (64, 8, 144, 24, 64, 132), "Exact-budget comparison")
    require(4*(3*10+7) == 148 > max(old_separated, 108), "Fixed nine-term lower bound")
    require(DELTA**4/DELTA <= DELTA and comparators(7) == (0, 4, 64, 12) and
            4*(10+7) == 68 > 64, "Seven-term scales and equal-budget improvement")
    print("Two disjoint ranges give 10 + 7 simple returns per quadrant at exactly seven terms.")
    print("Four disjoint ranges give 10 + 10 + 10 + 7 simple returns per quadrant at the specified rates.")


def collected_field(pinned, exponential_indices):
    """Collect all terms with the selected exponential rates and the fixed linear tail."""
    p, old_q = pinned
    wall = max(old_q)
    D = wall[0]//2
    require(wall == (2*D, 1+D) and D%2 == 0, "Common integer scale read from the old field")
    tail = next(pair for pair in old_q if pair[0] == D)
    g = old_q[tail]/T**3
    rates = [DELTA**j for j in exponential_indices] + [DELTA**4]
    require(g > 0 and rates[-1] == T**3 and all(D % (2*b.denominator) == 0 for b in rates),
            "All new dyadic powers lift evenly")
    q = collect([(wall[0], wall[1], -g*sum(rates)), (tail[0], tail[1], g*rates[-1])] +
                [(D+(D//b.denominator)*b.numerator, 1+D, g*b) for b in rates[:-1]])
    require((len(p), len(q)) == (4, len(rates)+1) and all(p.values()) and all(q.values()),
            "Fully collected supports")
    require(all(x%2 and y%2 == 0 for x, y in p) and
            all(x%2 == 0 and y%2 for x, y in q), "Four quadrant copies")
    for (x, y), coefficient in q.items():
        if x not in (D, 2*D):
            require((x-D)*g == coefficient*D and y == 1+D, "Read back the new horizontal rates")
    return p, q


def check_archive(export):
    pinned = eight_field()
    field = collected_field(pinned, (1, 2, 3))
    if export:
        with FIELD_PATH.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as out:
            for name, part in zip(("P", "Q"), field):
                out.write((name+"\n").encode("ascii"))
                for (x, y), c in sorted(part.items()):
                    out.write(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}\n".encode("ascii"))
    with gzip.open(FIELD_PATH, "rt", encoding="ascii") as src:
        for name, part in zip(("P", "Q"), field):
            require(src.readline().strip() == name, "Component in archive")
            for (x, y), c in sorted(part.items()):
                row = tuple(int(token, 16) for token in src.readline().split())
                require(row == (x, y, c.numerator, c.denominator), "Complete exponent and coefficient readback")
        require(src.read() == "", "End of nine-term field")
    bits = [i.bit_length() for part in field for key in part for i in key]
    digest = fingerprint(field)
    require(digest == EXPECTED_FIELD, "Pinned canonical field fingerprint")
    print(f"Nine collected rows with exponent lengths {min(bits)}..{max(bits)} bits. "
          f"Binary fingerprint {digest}")
    # Q is sorted by x power: tail, b3, b2, b1, wall. The seven-term
    # specialization omits two rows and adds their coefficients to the wall.
    p, q = field
    tail, b3, b2, b1, wall = sorted(q)
    reduced = {tail: q[tail], b1: q[b1], wall: q[wall]+q[b2]+q[b3]}
    seven = collected_field(pinned, (1,))
    require(seven == (p, reduced) and tuple(map(len, seven)) == (4, 3), "Seven-term archive decoding")
    digest = fingerprint(seven)
    require(digest == "9d37efdd151086913cbab28de9beaf45b03965520d03f3615dd4135df451da60",
            "Canonical seven-term fingerprint")
    seven_path = Path(__file__).resolve().parents[2] / "outputs" / "seven-field.hex.gz"
    with seven_path.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as out:
        for name, part in zip(("P", "Q"), seven):
            out.write((name + "\n").encode("ascii"))
            for (x, y), c in sorted(part.items()):
                out.write(f"{x:x} {y:x} {c.numerator:x} {c.denominator:x}\n".encode("ascii"))
    with gzip.open(seven_path, "rt", encoding="ascii") as src:
        for name, part in zip(("P", "Q"), (p, reduced)):
            require(src.readline().strip() == name, "Seven-term component header")
            for (x, y), c in sorted(part.items()):
                require(tuple(int(token, 16) for token in src.readline().split())
                        == (x, y, c.numerator, c.denominator), "Complete seven-term coefficient readback")
        require(src.read() == "", "End of seven-term field")
    print(f"Seven collected rows. Binary fingerprint {digest}")


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--field", action="store_true", help="recompute and read back the whole fixed field")
    group.add_argument("--export", action="store_true", help="regenerate the compressed field and read it back")
    args = parser.parse_args()
    if args.field or args.export:
        check_archive(args.export)
    else:
        check_cross_rank()
        check_structural_examples()
        check_nine_estimates()


if __name__ == "__main__":
    main()
