"""Exact degree and term ceilings and finite comparisons at common budgets."""
import csv
from fractions import Fraction
from hashlib import sha256
from math import isqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finite import endpoint_count, folded, stationary
from check_mechanical_support import source_counts


def closure(native):
    """Apply k-fold replication at degree k(d+1)-1 and degree monotonicity."""
    bound = native.copy()
    for degree, count in enumerate(native):
        if count:
            for k in range(2, len(native) // (degree + 1) + 1):
                target = k * (degree + 1) - 1
                bound[target] = max(bound[target], count * k * k)
    for degree in range(1, len(bound)):
        bound[degree] = max(bound[degree - 1], bound[degree])
    return bound


def native_families(limit):
    roots = {3: (1, 13), 4: (1, 20), 5: (2, 28), 6: (2, 35)}
    reversible_seeds = {4:28, 5:37, 6:53, 7:74, 8:96, 9:120, 10:142,
                        13:212, 17:384, 21:568, 31:1184, 35:1536, 39:1920, 43:2272}
    classical, binary, ternary, mechanical = ([0] * (limit+1) for _ in range(4))
    ranks = [0] * (limit+1)
    for degree in range(3, limit+1):
        if degree in roots:
            ranks[degree], classical[degree] = roots[degree]
        else:
            parent = (degree-1)//2
            rank = ranks[parent]
            increment = (4*rank+1)*parent - (2*rank if degree%2 else 0)
            ranks[degree] = 2*rank+1
            classical[degree] = 4*classical[parent] + increment
        budget = degree+5
        if budget >= 16:
            parent = budget//2
            binary[degree] = 4*(binary[parent-5]+(parent-3)*(parent-4))
    # Only the native Han--Li states reproduce along this tree.
    for degree, count in {2:4, **reversible_seeds}.items():
        classical[degree] = max(classical[degree], count)
    r, n = 2, 9
    while 4*n-5 <= limit:
        ternary[4*n-5] = 4*r*r*(n//3)**2 - (n-1)**2
        r, n = r+1, 3*n
    half = [0] * ((limit+1)//2+1)
    for n in range(2, len(half)):
        depth = n.bit_length()-1
        parent, digit = divmod(n, 2)
        loss = 0
        for level in range(depth):
            residue = parent % (1 << level)
            loss += residue*(residue+digit)
        births = depth*(n*n-digit)-4*(loss+(1<<depth)-1)
        half[n] = 4*half[parent]+births
    for degree in range(3, limit+1):
        mechanical[degree] = half[(degree+1)//2]
    for n in range(2, 101):
        assert half[n] == source_counts(n)
    return classical, binary, ternary, mechanical


def degree_comparison():
    with (ROOT/"inputs/finite-degree-table.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    high = max(int(row["degree"]) for row in rows)
    native = native_families(high)
    c, b, q, m = [closure(family) for family in native]
    # Canonical projection of the four reconstructed November columns. The ternary
    # family has no seed below degree 31, represented here by an empty bound 0.
    ledger = sha256()
    for d in range(3, 20001):
        ledger.update(f"{d}\t{c[d]}\t{b[d]}\t{q[d]}\t{m[d]}\n".encode("ascii"))
    assert ledger.hexdigest() == "da6cff395555f1fae4d35d5bd70d2188e5eafd673fdff207f03c3b9ef8628f4f"
    print(f"Four recomputed columns at all 19,998 degrees 3..20000: SHA-256 {ledger.hexdigest()}.")
    for row in rows:
        d = int(row["degree"])
        values = [c[d], b[d], q[d], m[d]]
        expected = [int(row[col] or 0) for col in ("classical","two_source","ternary","mechanical")]
        assert values == expected, (d, values, expected)
    assert len(rows) == 17 and high == 100000
    print("17 bundled integer rows matched. The four-column projection is an arithmetic check.")


def saturated_support_and_fixed_modes():
    """Check the exact-degree witness and fixed-stage integer allocations."""
    for cap in range(2, 2049):
        odd_slots = range(1, cap + 1, 2)
        even_slots = range(2, cap + 1, 2)
        terms = 2 * len(odd_slots) + len(even_slots)
        assert terms == (3 * cap + 1) // 2
        assert max(odd_slots[-1], even_slots[-1]) == cap
        assert (odd_slots[-1] == cap) == bool(cap % 2)
        assert 4 * (cap // 2) * ((cap + 1) // 2) <= cap**2
    print("Caps 2..2048: both odd supports and the positive even support reach exact degree "
          "and exactly ceil(3*cap/2) component positions, for both parities.")

    # For the diagonal use h=2**b with b a power of two, so the log/log
    # stage choice can be checked exactly at a family of rounding boundaries.
    def check_allocation(cap, terms, depth, p, d, x, group):
        scale = 1 << depth
        assert x >= 2 * group and x % (2 * group) == 0
        assert p >= 12 * scale and d >= 6 * scale and p + x + d <= terms
        assert max(2*p - 1, 2*x - 1, 2*d) <= cap
        for stage in range(depth):
            d_k = d >> stage
            length = (d_k + 1) // 2
            assert length == (d + (1 << stage)) >> (stage + 1)
            assert 2 * length - 1 <= d_k and 2 * length + 1 > d_k
            assert 2 * (d >> (stage + 1)) <= d_k

    for b in (16, 32, 64, 128, 256):
        h = 1 << b
        r = b - (b.bit_length() - 1) - 5
        group = 1 << (max(2, r) - 1).bit_length()
        x = 2 * group * ((h // 2) // (2 * group))
        p = 12 * (1 << r)
        d = min(h // 2, h + 1 - x - p)
        check_allocation(h, h + 1, r, p, d, x, group)
    for cap in range(1024, 4097):
        r = (cap // 24).bit_length() - 1
        group = 1 << (max(2, r) - 1).bit_length()
        x = 2 * group * ((cap // 2) // (2 * group))
        check_allocation(cap, (3 * cap + 1) // 2, r, cap // 2, cap // 2, x, group)
    print("Fixed-potential stage budgets, physical degrees, and all per-stage odd lengths checked "
          "on five exact diagonal log/log boundaries and all saturated caps 1024..4096.")


def term_comparison():
    first_elementary, first_stationary = None, None
    low_mech = [0, 0] + [source_counts(n) for n in range(2,101)]
    for terms in range(8, 20001):
        n = isqrt(terms//2)
        elementary = (terms-2)**2//4
        assert low_mech[n] < elementary
        if terms >= 48:
            count = folded(terms)["cycles"]
            if first_elementary is None and count > elementary:
                first_elementary = terms
            station = (terms-4)**2 - int(terms%2==0)
            if terms >= 512:
                assert count > elementary
            if terms >= 8192:
                assert count > station
            if first_stationary is None and count > station:
                first_stationary = terms
    assert (first_elementary, first_stationary) == (512,8192)
    assert low_mech[2:4] == [0,4]
    print("8..20000 exact terms: the direct floor-choice mechanical count is below the elementary count.")
    print("48..20000 exact terms: first strict folded improvement over elementary at 512, over stationary at 8192; both inequalities persist through the scanned endpoint.")
    allocation = folded(48)
    row = allocation["stages"][0]
    assert (allocation["p"], allocation["X"], allocation["d"], row["Y"], row["q"], row["alpha"]) == (24,12,12,2,2,4)
    pure_modes = (allocation["d"]+1)//2
    assert pure_modes == 6 and endpoint_count(2, 4, pure_modes) == 2
    assert row["Y"]*(allocation["X"]//row["q"])*endpoint_count(2,4,pure_modes) == 24
    assert row["used"] == row["B"] == 14 and allocation["cycles"] == 168 and allocation["D"] == 47
    terms, cap = 48, 47
    elementary = (terms-2)**2//4
    turn = stationary(terms)
    # Huzak--De Maesschalck, Theorem 1.1, with deg F=n and deg G=m.
    n = m = 12
    k, t = 22, 2
    seed = 2*((n-2)//2) + m//2
    term_ceiling = k + t*(max(n, m)+1)
    degree_ceiling = max(2*k-1, t*(max(n, m)+1)-1)
    assert (seed, t*k*seed, term_ceiling, degree_ceiling) == (16, 704, terms, 43)
    assert term_ceiling <= 3*degree_ceiling+2 and degree_ceiling <= cap
    assert (elementary, turn["cycles"], turn["D"]) == (529, 1935, 46)
    assert allocation["cycles"] < elementary < t*k*seed < turn["cycles"]
    print("N=48, cap 47, same fold allocation: fixed potential 24, joint variation 168 (degree 47).")
    print(f"Other fields under that budget: elementary {elementary}, replicated degree-12 seed "
          f"{t*k*seed}, stationary turn {turn['cycles']}.")
    print("Huzak--De Maesschalck Thm 1.1 gives 16 at bidegree (12,12). At k=22, t=2, "
          "pure-power filling to 48 terms is available within degree 43.")
    # q_j reaches its cap b by the start of the i=256 block.
    b = 2**128
    start_capped = 256*(256+7)//2
    assert b-7-start_capped+1 >= b//4
    partial = 4+sum(Fraction(4*m+9, 2**m) for m in range(1,9))
    lower = partial + Fraction(1,4)
    assert lower == 21+Fraction(15,256)
    loss = 7+Fraction(5,2)+Fraction(3*22,4)+Fraction(41,2)+Fraction(60,128)
    assert loss == Fraction(1503,32) < 50
    print(f"At b=2^128 distinct early and capped indices give sum(1/q_j) >= {lower} > 21. The retained normalized loss bound is {loss}.")
    print("At m asymptotic to c*Delta^2/log_2(Delta), the substituted recurrence and folded lower expression have coefficients c/4 and 2. They agree for c=8.")


if __name__ == "__main__":
    degree_comparison()
    term_comparison()
    saturated_support_and_fixed_modes()
