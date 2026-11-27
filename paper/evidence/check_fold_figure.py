"""Check the plotted preimages of P4(z) = 1/2 using exact angles and rational root counts."""
from fractions import Fraction
from pathlib import Path
import re

from sympy import Poly, QQ, Rational, chebyshevt, gamma, pi, simplify, sqrt, symbols


def check_roots(figure):
    # Read the actual PGF loop, including its degree-valued angles and both signs of z.
    loop = re.search(r"\\foreach\s+\\a\s+in\s+\{([0-9,\s]+)\}\s*\{(.*?)\n\s*\}",
                     figure, re.DOTALL)
    assert loop, "missing Figure 1 preimage loop"
    plotted = [int(a) for a in loop[1].split(",")]
    formula = re.search(r"\\pgfmathsetmacro\{\\xx\}\{sqrt\(\(1\+cos\(([0-9.]+)\*\\a\)"
                        r"/cos\(([0-9.]+)\)\)/2\)\}", loop[2])
    assert formula, "expected the inverse of c4*(2*z**2-1)"
    angle_unit, cutoff = (Fraction(v)/180 for v in formula.groups())
    assert (angle_unit, cutoff) == (Fraction(1,24), Fraction(1,16))
    assert sorted(re.findall(r"\\fill\s*\(([^,]+),0\)", loop[2])) == [r"-\xx", r"\xx"]

    # Write t=cos(theta). Since 0<theta<pi, cos(8*theta)=1/2 gives these eight residues.
    angles = [a for a in range(1,24) if (8*a*angle_unit) % 2 in (Fraction(1,3), Fraction(5,3))]
    assert angles == [1,5,7,11,13,17,19,23]
    inside = [a for a in angles if cutoff < a*angle_unit < 1-cutoff]
    assert sorted(plotted) == inside and len(inside) == 6
    assert [a for a in angles if a*angle_unit < cutoff] == [1]
    assert [a for a in angles if a*angle_unit > 1-cutoff] == [23]

    # Independent polynomial check. The largest zero of T8 is c4=cos(pi/16).
    t = symbols("t")
    outer = Poly(chebyshevt(8,t), t, domain=QQ)
    level = 2*outer - 1
    low, high = Rational(49,50), Rational(99,100)
    assert outer.count_roots(low, high) == 1 and outer.count_roots(high, 1) == 0
    assert outer.eval(low) != 0 and outer.eval(high) != 0
    cuts = (-1, -high, -low, low, high, 1)
    assert all(level.eval(x) != 0 for x in cuts)
    counts = [level.count_roots(a,b) for a,b in zip(cuts,cuts[1:])]
    assert counts == [1,0,6,0,1] and sum(counts) == level.degree()
    assert level.gcd(level.diff()).degree() == outer.gcd(outer.diff()).degree() == 0
    # T8 is even, T8(c4)=0, so t=+/-c4 never occurs at level 1/2.
    # Each t gives z^2=(1+t/c4)/2. The six middle t's lie strictly between -c4 and c4.
    assert all(power % 2 == 0 for (power,), coefficient in outer.terms())
    print("T8(t)=1/2: eight simple roots with angles a*pi/24 for a=1,5,7,11,13,17,19,23.")
    print("Sturm counts from -1 to 1 at the four internal cuts -99/100, -49/50, 49/50, 99/100: 1,0,6,0,1.")
    print("Plotted a=5,7,11,13,17,19 with both signs. P4(z)=1/2: 12 simple roots in (-1,1),"
          " 0 at its endpoints, 2 real exterior roots and 2 nonreal roots. Real-axis total: 14.")


def check_legend(figure):
    label = next(line for line in figure.splitlines() if r"\{z" in line and "P_4(z)" in line)
    caption = figure.split(r"\caption{",1)[1].split(r"\label{",1)[0]
    assert r"z\in(-1,1)" in label.replace(" ",""), "missing open interval on the plotted set"
    assert r"\((-1,1)\)" in caption, "missing open interval in the block-figure caption"
    print("Both the set label and the block-figure caption specify the open unit interval.")


def check_equal_saddles():
    x = symbols("x")
    f = Poly(chebyshevt(3, x), x, domain=QQ)
    w = f**2
    critical = f.diff().real_roots()
    assert len(f.real_roots()) == 3 and critical == [Rational(-1, 2), Rational(1, 2)]
    assert all(w.eval(c) == 1 for c in critical)
    # Consecutive pairs of simple turning points bound the sublevel intervals.
    inner, outer = [(w-level).count_roots()//2 for level in (Rational(1, 2), 2)]
    assert (inner, outer, inner+outer, 2*inner+outer) == (3, 1, 4, 7)
    print("T3 squared: both intervening saddles have height 1; three inner and one outer annulus, seven finite ends.")


def check_singular_taylor_coefficient():
    """A negative endpoint exponent in Lemma 2.1 on W=(x^2-1)^2."""
    x, u = symbols("x u", positive=True)
    potential = (x**2-1)**2
    exponent = Rational(3,4)
    beta = exponent-1
    assert potential.subs(x, sqrt(2)) == 1
    assert potential.diff(x).subs(x, sqrt(2)) != 0
    # On the right limiting interval use x=sqrt(2*u), 0<u<1.
    assert simplify(1-potential.subs(x, sqrt(2*u))-4*u*(1-u)) == 0
    assert simplify(sqrt(2*u).diff(u)-1/sqrt(2*u)) == 0
    a, b = beta+Rational(1,2), beta+1
    inner = simplify(exponent * 4**beta / sqrt(2) * gamma(a)*gamma(b)/gamma(a+b))
    assert simplify(inner-3*pi/(4*sqrt(2))) == 0
    assert simplify(2*inner-3*pi/(2*sqrt(2))) == 0
    print("For W=(x^2-1)^2 and gamma=3/4 the first Taylor coefficient "
          "is 3*pi/(4*sqrt(2)) on each inner interval and twice that on the outer interval.")


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1] / "tex/fold-block.tex"
    figure = source.read_text()
    check_roots(figure)
    check_legend(figure)
    check_equal_saddles()
    check_singular_taylor_coefficient()

