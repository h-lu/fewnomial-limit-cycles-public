"""Recover both nonnegative exponent tables independently and compare every coefficient."""
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = {6: "outputs/six-field.hex.gz", 7: "outputs/seven-field.hex.gz",
            8: "research/tail_interpolation/field.hex.gz", 9: "research/cross_visibility/field.hex.gz"}


def digits(row):
    return [int(s, 16) for s in row.split()]


def check():
    with gzip.open(ROOT / "outputs/primitive-factors.hex.gz", "rb") as report:
        assert report.readline().startswith(b"# hex:")
        for count, name in ARCHIVED.items():
            n, a, b, e, even_degree, primitive_degree = digits(report.readline())
            assert n == count
            for mode, i, j, expected in (("even", a//2*2, b//2*2, even_degree),
                                         ("primitive", a, b, primitive_degree)):
                with TemporaryFile() as moved:
                    subprocess.run([sys.executable, "-B", "evidence/check_primitive_fields.py",
                                    "--field", str(n), "--mode", mode], cwd=ROOT, stdout=moved, check=True)
                    moved.seek(0)
                    h = hashlib.sha256()
                    highest, rows = 0, 0
                    with gzip.open(ROOT / name, "rb") as original:
                        for old, new in zip(original, moved, strict=True):
                            h.update(new)
                            if old.strip() in (b"P", b"Q"):
                                assert old.strip() == new.strip()
                                continue
                            x, y, num, den = digits(new)
                            assert digits(old) == [x+i, y+j, num, den]
                            highest = max(highest, x+y)
                            rows += 1
                    assert highest == expected == e-i-j and rows == n
                print(f"N={n} {mode}: all {n} exponent pairs and rational coefficients recovered; SHA-256 {h.hexdigest()}")
        assert not report.readline()


if __name__ == "__main__":
    check()
