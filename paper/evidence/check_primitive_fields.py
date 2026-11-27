"""Primitive representatives of the archived polynomial fields (hexadecimal integers)."""
import argparse
import gzip
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {6: "outputs/six-field.hex.gz", 7: "outputs/seven-field.hex.gz",
         8: "research/tail_interpolation/field.hex.gz", 9: "research/cross_visibility/field.hex.gz"}


def read_field(path):
    parts = {}
    with gzip.open(path, "rt", encoding="ascii") as stream:
        for line in stream:
            line = line.strip()
            if line in ("P", "Q"):
                component = line
                parts[component] = []
            else:
                parts[component].append(tuple(int(item, 16) for item in line.split()))
    return parts


def write_field(field, a, b, stream):
    for component, rows in field.items():
        print(component, file=stream)
        for x, y, num, den in rows:
            print(*(format(i, "x") for i in (x-a, y-b, num, den)), file=stream)


def factor(field):
    rows = field["P"] + field["Q"]
    a = min(row[0] for row in rows)
    b = min(row[1] for row in rows)
    degree = max(x+y for x, y, _, _ in rows)
    # A Minkowski summand contributes an edge direction to its Newton polygon.
    # Every chord of P is steep; every chord of Q is shallow (including a segment Q).
    for part, steep in (("P", True), ("Q", False)):
        for (x, y, _, _), (u, v, _, _) in combinations(field[part], 2):
            if steep:
                assert abs(y-v) > abs(x-u)
            else:
                assert abs(y-v) < abs(x-u)
    for shift in ((a-a%2,b-b%2),(a,b)):
        moved = [[(x-shift[0], y-shift[1]) for x,y,_,_ in part] for part in field.values()]
        assert all(len(set(part)) == len(part) for part in moved)
        pairs = sum(moved, [])
        assert all(i >= 0 and j >= 0 for i,j in pairs)
        assert max(i+j for i,j in pairs) == degree-sum(shift)
    return a, b, degree


def check():
    exact = []
    previous = None
    for count, name in FILES.items():
        field = read_field(ROOT/name)
        assert (len(field["P"]),len(field["Q"])) == (4,count-4)
        a,b,d = factor(field)
        reduced = d-a-b
        assert (a%2,b%2) == (1,1)
        assert (d.bit_length(), reduced.bit_length()) == (
            (524615,524614) if count == 6 else (9106819,9106819))
        assert ((3*(a+b)>d) if count == 6 else (3*(a+b)<d))
        assert 333333*d <= 1000000*(a+b) < 333334*d
        if count >= 8:
            assert (a,b,d) == previous
        previous = (a,b,d)
        line = " ".join(format(v,"x") for v in (count,a,b,d,d-a-b+2,reduced))
        exact.append(line)
        digest = sha256(line.encode("ascii")).hexdigest()
        print(f"N={count}, raw/primitive degree bits {d.bit_length()}/{reduced.bit_length()}, "
              f"gcd exponents odd/odd, 3(a+b) {'>' if count==6 else '<'} raw degree, row SHA-256 {digest}")
    data = ("# hex: N min_x min_y raw_degree even_quotient_degree primitive_degree\n"
            + "\n".join(exact) + "\n").encode("ascii")
    out = ROOT/"outputs/primitive-factors.hex.gz"
    with out.open("wb") as raw:
        with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as packed:
            packed.write(data)
    print("Maximum even cancellation leaves xy. Full cancellation is coprime over C.")
    print("Orientation preserved for xy>0 and reversed for xy<0 by full cancellation.")
    print("N=10: parametric U3.11, horizontal Q and distinct y powers of P; degree E10-a-b.")
    print("The same term counts persist. The analytic quadrant conjugacies transfer the cycles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field",type=int,choices=FILES)
    parser.add_argument("--mode",choices=("even","primitive"),default="primitive")
    args = parser.parse_args()
    if args.field is None:
        check()
    else:
        field = read_field(ROOT/FILES[args.field])
        a,b,_ = factor(field)
        if args.mode == "even":
            a,b = a-a%2,b-b%2
        write_field(field,a,b,sys.stdout)
