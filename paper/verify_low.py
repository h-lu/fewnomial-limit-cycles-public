"""Recompute exact budgets, capped returns, published support data and fixed degrees."""
import gzip
import hashlib
import json
from fractions import Fraction
from math import comb
from pathlib import Path

import sympy as sp
from flint import arb, ctx
from finite import result
from research.separated_exponents.verify import vertical_system
from research.cross_visibility.verify import collected_field, eight_field

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TEX = ROOT / "tex"
A = json.loads((ROOT / "inputs/budgets.json").read_text(), parse_float=str)
B = json.loads((ROOT / "inputs/public.json").read_text())
def save(path, s):
    path.write_text(s.rstrip() + "\n", encoding="ascii")

def quartic_intervals(d, marks, panels=8192):
    y = sp.Symbol("y")
    j = d-1
    L = sum(comb(2*j+1,2*k)*y**(2*j+1-2*k)*(2-y*y)**k for k in range(j+1))
    coeff = [int(sp.expand(sp.diff(L,y)).coeff(y,2*k)) for k in range(d)]
    ctx.prec = 120
    rows = []
    halfpi = arb.pi()/2
    unit = arb(1)/(2*panels)
    segments = []
    for i in range(panels):
        theta = halfpi * arb((2*i+1)*unit,unit)
        segments.append((theta.sin()**2,theta.cos()**2))
    for raw in marks:
        E = arb(str(raw))
        t = E.sqrt()
        total = arb(0)
        for sn,cs in segments:
            v = (1+t)*sn
            D = arb(0)
            for c in reversed(coeff):
                D = D*v+c
            total += cs*(t-1+v).sqrt()*D
        total *= halfpi/panels
        assert total>0 or total<0, (d,raw,total)
        # Enlarge the ball to an easily checked rational interval.
        lo = int((1000 * total.lower().fmpq()).floor()) - 1
        hi = int((1000 * total.upper().fmpq()).ceil()) + 1
        assert (total-arb(lo)/1000)>0 and (arb(hi)/1000-total)>0
        assert lo*hi>0
        rows.append(dict(E=str(raw),lo_milli=lo,hi_milli=hi,sign=1 if lo>0 else -1))
    return rows

def focus_check():
    # Lagrange inversion for the even coefficients on the trace-zero surface.
    u,b=sp.symbols("u b")
    f=-sp.exp(2*u)+sp.exp(-u)
    V=sp.exp(u)-1-u-b*((sp.exp(3*u)-1)/3-u)
    A0=1-3*b
    def even(j,V0,A1,f0):
        ratio=sp.series(2*V0/(A1*u*u),u,0,2*j+1).removeO()
        return sp.factor(sp.series(f0*ratio**sp.Rational(-2*j-1,2),u,0,2*j+1).coeff(u,2*j))
    c2=even(1,V,A0,f)
    c4=even(2,V.subs(b,0),1,f)
    assert sp.simplify(c2-9*b/(3*b-1))==0 and c4==sp.Rational(1,3)
    return [str(c2),str(c4)]

def focus_support(k, lift):
    pp = {(1,2),(0,0)}|{(0,j+1) for j in range(2,k+1)}
    qq = {(1,1),(0,0)}
    if lift=="both":
        pp={(2*x,2*y+1) for x,y in pp}
        qq={(2*x+1,2*y) for x,y in qq}
    if lift=="x":
        pp={(2*x,y) for x,y in pp}
        qq={(2*x+1,y) for x,y in qq}
    return [len(pp),len(qq),max(x+y for x,y in pp|qq)]

def changes(rows):
    return sum(a!=b for a,b in zip(rows,rows[1:]))

def cell(xs):
    return r"\begin{tabular}[t]{@{}l@{}}" + r"\\ ".join("$"+x+"$" for x in xs) + r"\end{tabular}"
def entry(C,D,kind,cap,mark=""):
    return f"{C}{mark},\\,{D},\\,{kind},\\,{cap}"
def table_line(items):
    return " & ".join(items)+r" \\"
def run():
    OUT.mkdir(exist_ok=True)
    frows=[result(n) for n in A["fold_examples"]]
    save(OUT/"finite-stages.json",json.dumps(frows,indent=2))
    lines=[]
    for z in frows:
        f=z["fold"]
        terms = z["N"]
        if terms in (48, 256, 512, 1024, 4096, 8192, 32768, 65536, 524288, 1048576):
            values = (terms, f["r"], f["D"], f["cycles"], z["stationary"]["cycles"],
                      z["separate_strips"]["cycles"])
            lines.append(table_line([f"{value:,}" for value in values]))
    save(TEX/"fold-rows.tex","\n".join(lines))
    assert [(z["N"],z["fold"]["cycles"]) for z in frows if z["N"] in (48,257,1048576)]==[(48,168),(257,10496),(1048576,3718523125760)]
    for n in range(1,2049):
        z=result(n)
        u=z["uniform"]
        assert sum(u["support"])==n and u["D"]<n
        if z["fold"]:
            from finite import endpoint_count
            for st in z["fold"]["stages"]:
                if n<270:
                    assert st["B"]==max(endpoint_count(st["q"],st["alpha"],l) for l in range(st["L"]+1))
    quart={d:quartic_intervals(d,A["quartic_marks"][str(d)]) for d in (2,3,4)}
    assert [changes([row["sign"] for row in quart[d]]) for d in (2,3,4)]==[1,1,2]
    focus=focus_check()
    fields=eight_field()
    degrees=[]
    for field in (collected_field(fields,(1,)),fields,collected_field(fields,(1,2,3))):
        degree=max(sum(pair) for part in field for pair in part)
        assert sum(max(field[0],key=lambda pair:pair[1]))==degree
        degrees.append(degree)
    assert len(set(degrees))==1
    deg=degrees[0]
    hexdata=(format(deg,"x")+"\n").encode()
    with (OUT/"fixed-degree.hex.gz").open("wb") as stream:
        with gzip.GzipFile(filename="",fileobj=stream,mode="wb",mtime=0) as gz:
            gz.write(hexdata)
    save(TEX/"degree-bits.tex",f'{deg.bit_length():,}')
    wave={k:{kind:focus_support(k,kind) for kind in ("none","x","both")} for k in B["gasull_santana"]["focus_k"]}
    assert [wave[k]["none"][2] for k in wave]==[3,3,4,5,6]
    assert [wave[k]["both"][2] for k in wave]==[7,7,9,11,13]
    assert wave[3]["both"] == [4,2,9]
    x,y = sp.symbols("x y")
    delta,epsilon,lam = (sp.Rational(B["songling"][key]) for key in ("delta","epsilon","lambda"))
    songling_p = sp.Poly(lam*x-y-10*x**2+(5+delta)*x*y+y**2,x,y)
    songling_q = sp.Poly(x+x**2+(-25+8*epsilon-9*delta)*x*y,x,y)
    songling = [len(songling_p.terms()),len(songling_q.terms()),
                max(songling_p.total_degree(),songling_q.total_degree())]
    assert songling == [5,3,2]
    b=B["bbjt"]
    assert all(Fraction(c) for c in b["g"]) and all(sorted(map(Fraction,b[key]))==list(map(Fraction,b[key])) for key in ("small_h","large_h"))
    assert 4*changes(b["small_sign"])+2*changes(b["large_sign"])==24
    matrices={p:str(vertical_system(p)[2]) for p in (2,3)}
    assert matrices=={2:"-1/4762800",3:"-4/35153326333125"}
    specials={4:(1,3,"perturbed linear center"),5:(4,4,"one-axis focus")}
    for n,x,d in B["local"]["quartic"]:
        specials[n]=(x*(2*(d-1)+changes([a["sign"] for a in quart[d]])),max(3,2*x-1,2*d-1),"two wells")
    for n,x,d in B["local"]["turn"]:
        specials[n]=((2*x-1)*(2*d-3),max(2*d,2*x-1),"stationary, reduced degree")
    upper={}
    for n,s,d in B["local"]["center"]:
        upper[n]=(4*s*d,"B_"+str(n),"center, clearing parameter")
    for n,p,k in B["local"]["separated"]:
        assert p+k+3==n
        upper[n]=(8*k*((9*p-2)//4),"E_"+str(n),"separated powers")
    for n,c in A["fixed"]:
        upper[n]=(c,"D_*","fixed vertical interpolant")
    publications={
      4:[(12,"S","O","?"),(4,7,"H","-"),(1,3,"H","+")],
      5:[(12,"S","O","?"),(8,7,"I","-"),(3,3,"I","+"),(2,3,"I","+")],
      6:[(12,"S","O","?"),(12,9,"I","-"),(3,4,"I","+")],
      7:[(16,11,"I","-"),(4,5,"I","+")],8:[(20,13,"I","-"),(5,6,"I","+"),(4,2,"H","+")],
      9:[(24,4,"I","+"),(24,"G_3","O","?"),(13,3,"O","+")],10:[(32,"G_4","O","?")]}
    attributed = {6:[(4,3,"I","+")],7:[(5,3,"I","+")]}
    persistence = {10:[(24,4,"I","+")]}
    assert [publications[n][0][0] for n in range(4,11)]==B["gasull_santana"]["unrestricted"]
    two_square_sources = {n: (4*(n-3), wave[n-3]["both"][2], "H", "-") for n in (4,5,6)}
    primitive_degree = {"E_6": r"E_6^\circ", "D_*": r"D_*^\circ", "E_10": r"E_{10}^\circ"}
    specialized = {}
    contributions = []
    for n in range(4, 11):
        count, degree, _ = specials[n]
        capped = (count, degree, "H", "+")
        if n <= 6:
            k = n-3
            one_square = (2*k, wave[k]["x"][2], "H", "+" if n >= 5 else "-")
            two_squares = (4*k, wave[k]["both"][2], "H", "-")
            power_law = (8, "S_L", "H", "-")
            refinements = [one_square, two_squares, power_law]
        else:
            refinements = []
        if n >= 6:
            count_out, cleared, _ = upper[n]
            large = (count_out, primitive_degree[cleared], "H", "-")
            refinements.append(large)
        specialized[n] = list(dict.fromkeys([capped] + refinements))
        general = result(n)
        article = general["uniform"]
        row = [str(n), str(general["base"]["cycles"]), str(article["cycles"]), str(article["D"])]
        extra = (str(count), str(degree)) if 5 <= n <= 8 else ("---", "---")
        row.extend(extra)
        capped_pub = ([(c_, d_, "") for c_, d_, _, cap_ in publications[n] if cap_ == "+"]
                      + [(c_, d_, "^{a}") for c_, d_, _, cap_ in attributed.get(n, []) if cap_ == "+"]
                      + [(c_, d_, "^{p}") for c_, d_, _, cap_ in persistence.get(n, []) if cap_ == "+"])
        best_c, best_d, best_mark = max(capped_pub, key=lambda t: (t[0], -t[1]))
        row.append(f"${best_c}{best_mark}$ ({best_d})" if best_mark else f"{best_c} ({best_d})")
        contributions.append(table_line(row))
    save(TEX/"contribution-rows.tex", "\n".join(contributions))
    details=[]
    lines=[]
    for n in A["low"]:
        u=result(n)["uniform"]
        if n<=3:
            pub=spc=[(0,u["D"],r"\circ","+")]
            which="zero field count"
        else:
            pub=publications[n]
            c,dd,which=specials[n]
            spc=specialized[n]
        uniform=(u["cycles"],u["D"],"H" if n>=4 else r"\circ","+")
        lit = [entry(*row, mark="^{f}" if n == 4 and row == (4,7,"H","-") else "") for row in pub]
        lit.extend(entry(*row,mark="^{a}") for row in attributed.get(n,[]))
        lit.extend(entry(*row,mark="^{p}") for row in persistence.get(n,[]))
        if n >= 4:
            lines.append(r"\addlinespace[2pt]")
        explicit = [entry(*x, mark=r"^{\star}" if x == two_square_sources.get(n) else "") for x in spc]
        lines.append(table_line([str(n),cell(lit),cell([entry(*uniform)]),cell(explicit)]))
        details.append(dict(N=n,published=pub,attributed_exact_N=attributed.get(n,[]),
                            from_persistence=persistence.get(n,[]),uniform=uniform,
                            fold="inapplicable",specialized=spc,capped_method=which,
                            unrestricted_method=("two-coordinate focus and four-quadrant power law" if n in (4,5) else upper[n][2]) if n>=4 else which))
    save(TEX/"low-rows.tex","\n".join(lines))
    audit=dict(low=details,quartic_intervals=quart,focus_coefficients=focus,focus_support_and_degree=wave,fixed_degree_binary_digits=deg.bit_length(),fixed_degree_hex_sha256=hashlib.sha256(hexdata).hexdigest(),separated_determinants=matrices,bbjt=dict(small_changes=5,large_changes=2,cycles=24))
    audit["songling_support_and_degree"] = songling
    save(OUT/"low-N.json",json.dumps(audit,indent=2))
    save(TEX/"quartic-rows.tex","\n".join(table_line([str(d),row["E"],str(row["lo_milli"]),str(row["hi_milli"])]) for d in quart for row in quart[d]))
    print("N=4..10 uniform", [result(n)["uniform"]["cycles"] for n in range(4,11)])
    print("N=4..10 capped specialized",[specials[n][0] for n in range(4,11)])
    print("N=4..10 unrestricted specialized",[upper[n][0] for n in range(4,11)])
    print("Quartic signed intervals",quart)
    print("Fixed integer degree:",deg.bit_length(),"binary digits, hexadecimal-text SHA-256",audit["fixed_degree_hex_sha256"])
    print("Finite support and cap checked at N=1..2048. Last-period maxima checked against every prefix at N=48..269.")
if __name__=="__main__":
    run()
