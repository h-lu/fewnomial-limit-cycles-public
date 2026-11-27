"""Published small-support fields, focus jets, product budgets, and uniform loss constants."""
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import sys
import sympy as sp
from flint import ctx, fmpq, fmpq_mat, fmpq_series

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from finite import classical_seed, folded, separate_strips


def focus_jet(j, b, c):
    # Coefficient of (-exp(2u)+exp(-u)) (2V/(A u^2))^(-(2j+1)/2)
    A = 1-3*b-4*c
    q = -sp.Rational(2*j+1,2)
    order = 2*j
    R = [sp.Integer(1)] + [2*(1-b*3**(i+1)-c*4**(i+1))/(A*factorial(i+2))
                            for i in range(1,order+1)]
    power = [sp.Integer(1)]
    for m in range(1,order+1):
        power.append(sp.factor(sum(((q+1)*i-m)*R[i]*power[m-i]
                                   for i in range(1,m+1))/m))
    return sp.factor(sum(((-1)**i-2**i)*power[order-i]/factorial(i) for i in range(order+1)))


def native_focus_data(coefficients):
    """Exact trace-zero damping and its parameter derivatives in the S2.1 chart."""
    z = fmpq_series([0, 1])
    terms = tuple(enumerate(coefficients, start=3))
    stiffness = 1 - sum(p*a for p, a in terms)
    potential = z.exp() - 1 - z - sum(
        a*(((p*z).exp()-1)/p-z) for p, a in terms
    )
    chart = z*(2*potential/(stiffness*z*z)).sqrt()
    u = chart.reversion()
    highest = 2*(len(terms)+1)
    assert chart(u).prec > highest and all(chart(u)[i] == z[i] for i in range(highest+1))
    f = (-u).exp() - (2*u).exp()
    damping = f*u.derivative()
    gradient = u.exp() - 1 - sum(a*((p*u).exp()-1) for p, a in terms)
    columns = []
    for p, _ in terms:
        # Differentiate V(u)=A*z^2/2 at fixed z, then f(u)*du/dz.
        du = (((p*u).exp()-1)/p-u-p*z*z/2)/gradient
        columns.append((f*du).derivative())
    assert damping.prec > highest and min(col.prec for col in columns) > highest-2
    jacobian = fmpq_mat([[col[2*j] for col in columns] for j in range(1, len(terms)+1)])
    return [damping[2*j] for j in range(1, len(terms)+2)], jacobian.det()


def published_focus_comparison():
    # The support is unchanged under the small unfoldings in the proof of GS Theorem 2.
    old_cap = ctx.cap
    try:
        ctx.cap = 22
        four = [fmpq(-42,109), fmpq(31,109), fmpq(-6,109)]
        tail = [fmpq(17,11), fmpq(-6,11), fmpq(7,99)]
        five = [fmpq(-161,99), *tail]
        assert native_focus_data(four) == ([0, 0, 0, fmpq(28,423)], fmpq(-282316322,2803221))
        assert native_focus_data([fmpq(-161,17), *tail])[0][0] == fmpq(19803,7162)
        assert native_focus_data(five) == ([0, 0, 0, 0, fmpq(-1,27)], fmpq(33,2))
        for coefficients in (four, five):
            assert all(coefficients) and sum(coefficients)-1 != 0
    finally:
        ctx.cap = old_cap
    print("GS (21), seven terms, degree 5: c2=c4=c6=0, c8=28/423, det=-282316322/2803221.")
    print("GS (21), eight terms, degree 6: the arXiv v3 value a2=-161/17 gives c2=19803/7162.")
    print("With a2=-161/99 and the other entries fixed: c2=c4=c6=c8=0, c10=-1/27, det=33/2.")
    print("The coefficients describe the organizing focus. The cycle counts concern nearby unfolding parameters.")


def low_support():
    b,c = sp.symbols("b c")
    j2,j4 = focus_jet(1,b,c),focus_jet(2,b,c)
    assert sp.cancel(j2-9*(b+2*c)/(3*b+4*c-1)) == 0
    assert sp.cancel(j4.subs(b,-2*c)+(19*c-1)/(3*(2*c+1))) == 0
    coefficients = {b: -sp.Rational(2,19), c: sp.Rational(1,19)}
    assert sp.factor(sp.Matrix([j2,j4]).jacobian((b,c)).det().subs(coefficients)) == sp.Rational(6859,147)
    assert focus_jet(3,coefficients[b],coefficients[c]) == -sp.Rational(4,21)
    print("At (a1,a2,a3)=(-1,-2/19,1/19): A=21/19, c2=c4=0, det=6859/147, c6=-4/21.")
    print("Gasull-Santana (21),(22). P and Q occupied exponents after collection:")
    x,y = sp.symbols("x y")
    for k in (1,2,3):
        aa = sp.symbols(f"a1:{k+1}")
        P = aa[0]*x*y**2 + sum(aa[j-1]*y**(j+1) for j in range(2,k+1)) - sum(aa)
        Q = x*y-1
        for label,p,q,want in (("native",P,Q,[3,3,4]),
                                 ("x-square",P.subs(x,x*x),2*x*Q.subs(x,x*x),[4,4,4]),
                                 ("both",y*P.subs({x:x*x,y:y*y}),x*Q.subs({x:x*x,y:y*y}),[7,7,9])):
            p,q = sp.Poly(p,x,y),sp.Poly(q,x,y)
            support=[z.monoms() for z in (p,q)]
            degree=max(z.total_degree() for z in (p,q))
            assert list(map(len,support))==[k+1,2] and degree==want[k-1]
            assert p.gcd(q).total_degree()==0
            print(f"N={k+3} {label}  D={degree}  P={support[0]}  Q={support[1]}  gcd=1")
    print("GS Theorem 2: 12 isolated at each N=4,5,6. For the power-law route, two local simple cycles per quadrant.")
    # Exponent algebra for the rational Boros-Hofbauer realization. No scale is fixed here.
    L,mu,delta=sp.symbols("L mu delta",positive=True)
    P=[(20*L+1,0),((12-mu)*L+1,(35-delta)*L)]
    Q=[(30*L,20*L+1),(0,28*L+1)]
    def difference(rows): return tuple(rows[1][i]-rows[0][i] for i in (0,1))
    u,v=difference(P),difference(Q)
    cross=sp.expand(u[0]*v[1]-u[1]*v[0])
    assert sp.expand(cross/L**2)==986-8*mu-30*delta
    assert sum(Q[0])==50*L+1
    print("BH (22)-(24): primitive two-binomial lift, determinant/L^2=986-8*mu-30*delta; D=50L+1.")
    print("This degree formula is parametric; the return argument chooses the rational exponents.")


def proposition_one_degrees():
    # For r=2 the proof of GS Theorem 2 uses n=3 and n=4. Write the
    # strict integer gaps as 1+u_i. The i-th mixed degree is 2n+1+2*sum(u_j, j<=i).
    x, y = sp.symbols("x y")
    q = sp.Poly(x*(x**2-1)*(x**2-4), x, y)
    for n in (3, 4):
        gaps = sp.symbols(f"u1:{n+1}", nonnegative=True, integer=True)
        exponents = [i + sum(gaps[:i]) for i in range(n+1)]
        degrees = [2*(n-i)+1 + 2*m for i,m in enumerate(exponents)]
        assert all(sp.expand(degrees[i]-degrees[i-1]) == 2*gaps[i-1] for i in range(1,n+1))
        assert sp.expand(degrees[-1] - (2*exponents[-1]+1)) == 0
        for sample in (tuple(range(n+1)), tuple(i*(i+3)//2 for i in range(n+1))):
            p = sp.Poly(y-y**3+sum((-1)**i*x**(2*(n-i)+1)*y**(2*m)
                                   for i,m in enumerate(sample)), x,y)
            assert (len(p.terms()), len(q.terms())) == (n+3, 3)
            assert p.gcd(q).total_degree() == 0
            assert max(p.total_degree(),q.total_degree()) == 2*sample[-1]+1
        print(f"GS Prop. 1 r=2, n={n}: N={n+6}, original sign count={8*n}, D=2*m_{n}+1 >= {2*n+1}.")
    print("The equalities m_i=i are necessary for these linear caps; the sample supports do not certify their return signs.")


def capped_and_center_fields():
    from research.verify_term_bounds import expanded_field

    x,y,A,b,c,mu = sp.symbols("x y A b c mu")
    fields = ((2*y, -2*x+mu*(-y+4*y**3/3),4,3),
              (b*x**2*y**2+c*y**3-b-c,2*x*(x**2*y-1),5,4),
              (4*y*(y**2-1),-sp.diff(A**2*(2*x**2-1)**2,x)+mu*(b*y+c*y**3),6,3))
    for P,Q,n,d in fields:
        p,q = (sp.Poly(z,x,y) for z in (P,Q))
        assert len(p.terms())+len(q.terms())==n and max(z.total_degree() for z in (p,q))==d
        assert p.gcd(q).total_degree()==0
        print(f"Capped table N={n}, D={d}: P={p.monoms()} Q={q.monoms()}; primitive.")

    for s,d in ((1,1),(1,2),(1,3),(2,3)):
        n=2*s+d+1
        p,q=expanded_field(s,d,10**20)
        keys=list(p)+list(q)
        mx=min(x for x,y in keys)
        my=min(y for x,y in keys)
        (x0,y0),(x1,y1)=sorted(p,key=lambda a:a[1])
        # Direction along the torus binomial P=0. Directly test that the top
        # exponent of the restricted Q is unique for every root of unity.
        orders=[(y1-y0)*x+(x0-x1)*y for x,y in q]
        assert orders.count(max(orders))==1
        raw=max(x+y for x,y in keys)
        even=raw-(mx//2*2)-(my//2*2)
        primitive=raw-mx-my
        assert all(i>=mx and j>=my for i,j in keys)
        assert len(keys)==n and all(p.values()) and all(q.values())
        if n==8:
            assert raw==1+7*192*10**300
        print(f"U2 N={n}, s={s}, d={d}, exponent scale input 10^20:"
              f" gcd minima ({mx},{my}); degrees {raw}, {even}, {primitive}.")
    print("U2 algebra samples test the displayed fields; the general cycle proof chooses a sufficiently small scale.")


def strip_budgets():
    # Expand the force at actual rational shifts; no cycle claim uses these samples.
    y = sp.Symbol("y")
    potential = sp.Poly(y**2, y)
    for level, shift in enumerate((-3, -10, -20, -50, -100), start=1):
        folded_potential = potential.compose(sp.Poly(y**2 + shift, y))
        force = folded_potential.diff()
        assert folded_potential.degree() == 2**(level + 1)
        assert {a for (a,), _ in force.terms()} == set(range(1, 2**(level + 1), 2))
        derivative = potential
        for j in range(1, potential.degree() + 1):
            derivative = derivative.diff()
            assert folded_potential.nth(2*j) * factorial(j) == derivative.eval(shift)
        potential = folded_potential
    print("Five exact quadratic compositions: force sizes 2, 4, 8, 16, 32; all odd coefficients checked.")

    tested = 0
    for delta in range(16, 2500):
        k = (delta + 1) // 2
        for is_diag, terms, residual in (
            (True, delta + 1, (delta + 2) // 2),
            (False, (3 * delta + 1) // 2, delta),
        ):
            assert terms - k == residual
            for rounding_parameter in (4, 7, 13):
                denom = rounding_parameter + 2 if is_diag else rounding_parameter + 1
                B = rounding_parameter * residual // denom
                if B < 2 * rounding_parameter:
                    continue
                p = (B // rounding_parameter).bit_length() - 1
                b = B // (1 << p)
                s = b * (1 << p)
                assert rounding_parameter <= b < 2 * rounding_parameter and 0 <= B - s < (1 << p) <= B // rounding_parameter
                count = 0
                for stage in range(p):
                    count = 2 * count + b * (1 << stage) - 1
                if is_diag:
                    force_size = 1 << (p + 1)
                    assert force_size + s - 1 + k < terms <= 2 * delta + k
                    assert max(2 * force_size - 1, 2 * s - 2, 2 * k - 1) <= delta
                    assert 2 * count == p * s - force_size + 2
                else:
                    force_size = 1 << p
                    assert force_size + s - 1 + k < terms <= 2 * delta + k
                    assert max(2 * force_size - 1, s - 1, 2 * k - 1) <= delta
                    assert count == p * s // 2 - force_size + 1
                tested += 1
    print(f"{tested} allocations at 16 <= Delta <= 2499, rounding parameter Lambda=4,7,13, for both target pairs when p >= 1.")
    print("Exact fold degree, pre-filling term ceiling, filling capacity and bracket recurrence pass.")


def budget_checks():
    strip_budgets()
    assert F(10)+3*22+4*F(41,2)==158
    assert F(12)+48+F(3,2)+72+4<=138
    loss=8+1+F(158,4)+F(138,256)
    assert loss==F(6277,128)<50
    print("Uniform large-N loss coefficient from the proof:",loss,"< 50.")
    # Independent exhaustive prefix maxima, precomputed once per (q,alpha).
    # Extend one value at a time so every length participates in the comparison.
    prefix = {}
    def best(q, alpha, length):
        values = prefix.setdefault((q, alpha), [0])
        while len(values) <= length:
            ell = len(values)
            b = (2*ell-1+alpha)//(2*alpha)
            c = (2*ell-2)//alpha
            free = ell - 2*(q-1)*(b+1) - 1
            contributions = (q*max(free,0), (2*q-2)*max(free-b,0), (q-1)*max(free-c,0))
            values.append(max(values[-1], sum(contributions)))
        return values[length]

    def independent_count(n):
        depth = max(1, n.bit_length()-8)
        group = 2
        while group < depth:
            group *= 2
        block_ends = [0]
        while block_ends[-1] + len(block_ends)+3 <= depth:
            block_ends.append(block_ends[-1] + len(block_ends)+3)
        x_limit = min(n//2, n-18*(2**depth))
        x = x_limit - x_limit%(2*group)
        p = max(12*2**depth, (n-x)//2)
        d = n-x-p
        assert max(2*x-1, 2*p-1, 2*d) < n and d >= 6*2**depth
        total = 0
        for j in range(1, depth+1):
            k = depth-j
            order = 2**max(1, sum(end <= j for end in block_ends[1:]))
            q = min(group, max(2, 2**((order.bit_length()-1)//2)))
            rows = 2**(k+1)
            for start, end in zip(block_ends, block_ends[1:]):
                if j <= start:
                    block = 2**(end-start)
                    assert rows % block == 0
                    rows = rows//block*(block-4)
            available = p//2**(k+1) + (d//2**k+1)//2
            total += rows*(x//q)*best(q, 2*order, max(0,available-2*order))
        return total

    for n in range(48,65537):
        f=folded(n)["cycles"]
        assert f == independent_count(n), n
        elementary=(n-2)**2//4
        stationary=(n-4)**2-int(n%2==0)
        assert (f<elementary if n<512 else f>elementary)
        assert (f<stationary if n<8192 else f>stationary)
    # This comparison is only on N=2^j. Independent prefix maxima give F.
    # At a fixed strip depth each extra initial mode increases c_i, so check
    # the last feasible width and the next one, then iterate from c_0=0.
    grid = []
    for exp in range(6, 21):
        n = 2**exp
        k = n // 2
        candidates = []
        for depth in range(1, exp):
            scale = 2**depth
            if 4*scale-1 > n-1:
                break
            def fits(width):
                size = scale * width
                return (2*scale+size-1+k <= n
                        and max(4*scale-1, 2*size-2, 2*k-1) <= n-1)
            lo, hi = 0, n
            while lo+1 < hi:
                mid = (lo+hi)//2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid
            if lo < 4:
                continue
            assert fits(lo) and not fits(lo+1)
            count, slots = 0, lo
            for _ in range(depth):
                count = 2*count+slots-1
                slots *= 2
            candidates.append((2*k*count, depth, lo))
        strips, depth, width = max(candidates)
        seed_degree = (n-1)//2
        copies = n-seed_degree-2
        seed_count = -((48-7*seed_degree)//2)
        classical = 2*copies*seed_count
        f = independent_count(n)
        assert f == folded(n)["cycles"]
        witness = separate_strips(n)
        assert (strips, depth, width) == (witness["cycles"], witness["p"], witness["b"])
        assert classical == classical_seed(n)["cycles"]
        grid.append((exp, n, f, strips, classical, depth, width))
    assert next(j for j, n, f, t, c, *_ in grid if f > c) == 16
    assert next(j for j, n, f, t, c, *_ in grid if f > t) == 20
    assert all(f != t and f != c for _, _, f, t, c, *_ in grid)
    path = ROOT / "outputs/dyadic-comparisons.tsv"
    path.write_text("j\tN\tfolded\tseparate_strips\tclassical_preprint\tstrip_depth\tinitial_modes\n"
                    + "".join("\t".join(map(str, row))+"\n" for row in grid))
    print("Dyadic grid j=6..20: first sampled F>C at j=16 and F>T at j=20. "
          "T fixes k=floor(N/2); C uses the classical preprint seed.")
    assert F(1061,1024)*(1-F(1,2**18)) > 1
    row = folded(48)["stages"][0]
    ell = row["used"]
    b, c = (ell+1)//4, (ell-1)//2
    j = ell-2*(b+1)-1
    assert (b,c,j) == (3,6,5) and (2*j,2*(j-b),max(0,j-c)) == (10,4,0)
    assert [((t+4)-2*((t+5)//4+1)-1)-((t+3)//2) for t in range(4)] == [-2,-2,-1,-3]
    first_flat_row = folded(65536)["stages"][0]
    length, group, alpha = (first_flat_row[key] for key in ("used", "q", "alpha"))
    b = (2*length-1+alpha)//(2*alpha)
    c = (2*length-2)//alpha
    pivots = length-2*(group-1)*(b+1)-1
    assert (length, group, alpha, pivots, c, pivots-c) == (16376, 2, 8, 12279, 4093, 8186)
    # The first occurrences are at a depth boundary; no construction of
    # exponentially many samples is needed to inspect one stage schedule.
    for n, flat, wells in ((2**16-1,4,2),(2**16,8,2),(2**29-1,16,2),(2**29,32,4)):
        stages = folded(n)["stages"]
        assert max(stage["alpha"] for stage in stages) == flat
        assert max(stage["q"] for stage in stages) == wells
    print("All 65,489 finite counts agree with independent exhaustive-prefix maxima; "
          "the analytic tail factor at N >= 32768 exceeds one.")
    print("N=48 per cluster: minima 10, incoming 4, outgoing 0. "
          "First alpha=8 at N=65536; first q=4 at N=2^29.")
    print("Exact scan 48..65536: first crossings of elementary/stationary at 512/8192; strict signs throughout this range.")


if __name__=="__main__":
    low_support()
    published_focus_comparison()
    proposition_one_degrees()
    capped_and_center_fields()
    budget_checks()
