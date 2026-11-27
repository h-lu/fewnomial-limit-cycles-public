"""A fully specified six-term specialization of the separated-scale proof."""
from fractions import Fraction as Q
from pathlib import Path
import gzip
import hashlib
from research.separated_exponents.verify import density, expanded_field, vertical_system
from research.separated_exponents.witness import DELTA,T,EPSILON,check_curvature,check_roots,check_transfers,fingerprint

def run():
    c=density(5,Q(1,16))
    check_curvature(c)
    check_roots(c)
    check_transfers()
    matrix,inv,det=vertical_system(2)
    assert det==Q(-1,4762800) and max(sum(map(abs,row)) for row in inv)==Q(91804,5)
    field=expanded_field(1,2,DELTA,T,EPSILON,c,offset=10**6)
    assert [len(p) for p in field]==[4,2]
    deg=max(sum(ij) for part in field for ij in part)
    assert deg>5 and max(sum(ij) for ij in field[0])==deg
    data=(format(deg,"x")+"\n").encode()
    out=Path(__file__).resolve().parent/"outputs"
    for name,raw in (("six-degree.hex.gz",data),("six-field.hex.gz",None)):
        with (out/name).open("wb") as f, gzip.GzipFile(filename="",fileobj=f,mode="wb",mtime=0) as z:
            if raw is not None:
                z.write(raw)
            else:
                for key,part in zip(("P","Q"),field):
                    z.write((key+"\n").encode("ascii"))
                    for (x,y),coeff in sorted(part.items()):
                        z.write(f"{x:x} {y:x} {coeff.numerator:x} {coeff.denominator:x}\n".encode("ascii"))
    print("Six fixed terms, one exponential range in each of four quadrants: 32 hyperbolic cycles.")
    print("8 brackets, endpoint margin >2^-465; slope >2^-136. Finite transfer checks passed.")
    print("Actual degree:",deg.bit_length(),"binary digits. Hexadecimal-text SHA-256:",hashlib.sha256(data).hexdigest())
    print("Canonical binary field SHA-256:",fingerprint(field))
    (out.parent/"tex"/"six-bits.tex").write_text(f"{deg.bit_length():,}\n")
if __name__=="__main__":
    run()
