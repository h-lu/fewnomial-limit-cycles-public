"""Read the delivered field and bracket tables without importing their generators."""
import csv
from fractions import Fraction
import gzip
from hashlib import sha256
from math import gcd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIELDS = [
    ("outputs/six-field.hex.gz", (4, 2), "six-degree.hex.gz", 524615,
     "0a7882af5006900b2b29a88c0d60243c05f025a66e9984e7cd66a854b568869a"),
    ("outputs/seven-field.hex.gz", (4, 3), "fixed-degree.hex.gz", 9106819,
     "9d37efdd151086913cbab28de9beaf45b03965520d03f3615dd4135df451da60"),
    ("research/tail_interpolation/field.hex.gz", (4, 4), "fixed-degree.hex.gz", 9106819,
     "01fb565a9b281cd37b0e97a6218c4a120894bc65f1164f5f7fc275b7663d3b83"),
    ("research/cross_visibility/field.hex.gz", (4, 5), "fixed-degree.hex.gz", 9106819,
     "56371e8a3da6793d1426aca2dd18121a48e430566afe58d18dde4ef381f84c8e"),
]


def check_field(path, sizes, degree_file, bits, expected_digest):
    digest = sha256()
    degree = 0
    with gzip.open(ROOT / path, "rb") as stream:
        for component, count in zip((b"P", b"Q"), sizes):
            assert stream.readline() == component + b"\n"
            digest.update(component)
            previous = (-1, -1)
            for _ in range(count):
                x, y, numerator, denominator = map(
                    lambda token: int(token, 16), stream.readline().split()
                )
                assert x >= 0 and y >= 0 and previous < (x, y)
                assert numerator and denominator > 0 and gcd(numerator, denominator) == 1
                previous = (x, y)
                degree = max(degree, x + y)
                for integer in (x, y, numerator, denominator):
                    size = max(1, (integer.bit_length() + 7) // 8)
                    digest.update(bytes([integer < 0]))
                    digest.update(size.to_bytes(8, "big"))
                    digest.update(abs(integer).to_bytes(size, "big"))
        assert stream.read() == b""
    with gzip.open(OUT / degree_file, "rb") as stream:
        assert degree == int(stream.read(), 16) and degree.bit_length() == bits
    assert digest.hexdigest() == expected_digest
    print(f"{path}: {sizes}, actual degree {bits} bits, canonical SHA-256 {expected_digest}")


def records(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader((s for s in stream if not s.startswith("#")), delimiter="\t"))


def check_brackets():
    rows = records("783-brackets.tsv")
    assert {row["sign"] for row in rows} == {"1", "-1"}
    exponents = [0, 31, 35, 39, 43, 47, 51, 55, 58, 59, 63, 67, 71, 75, 79]
    values = []
    for sign, powers in (("1", exponents), ("-1", exponents[1:])):
        selected = [row for row in rows if row["sign"] == sign]
        assert len(selected) == len(powers) - 1
        previous = Fraction(1, 2**10998)
        for row, e, f in zip(selected, powers, powers[1:]):
            assert (int(row["e"]), int(row["f"])) == (e, f)
            scale = Fraction(2) ** int(row["shift"])
            left = int(row["left_mantissa"]) * scale
            right = int(row["right_mantissa"]) * scale
            assert previous < left < right < Fraction(1, 2**2002)
            previous = right
            lo, hi, slope = [Fraction(int(row[col]), 2**20)
                             for col in ("left_lower", "right_upper", "slope_upper")]
            assert lo > Fraction(1, 8) and hi < -Fraction(1, 8) and slope < -Fraction(f-e, 2)
            values.append((lo, hi, slope))
    margins = min(v[0] for v in values), max(v[1] for v in values), max(v[2] for v in values)
    print("27 model brackets (14 positive, 13 negative). Weakest left, right and slope bounds:",
          ", ".join(map(str, margins)))
    quadrature = records("783-quadrature.tsv")
    expected = {(Fraction(sign, 65536), e) for sign in (-1, 1) for e in exponents}
    assert len(quadrature) == len(expected)
    assert {(Fraction(row["left_turn"]), int(row["exponent"])) for row in quadrature} == expected
    errors = [Fraction(int(row["error_upper"]), 2**36) for row in quadrature]
    assert min(errors) >= 0 and max(errors) < Fraction(1, 2**12)
    print(f"30 auxiliary integral enclosures. Largest exported relative error bound {max(errors)}.")


if __name__ == "__main__":
    for field in FIELDS:
        check_field(*field)
    check_brackets()
