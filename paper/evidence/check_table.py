"""Independent exhaustive-prefix evaluation of the printed folded allocations and finite comparison families."""
import json
from math import prod
from pathlib import Path

root = Path(__file__).resolve().parents[1]
# Printed folded allocations and additional rounding boundaries. N=32 is below the folded threshold.
fold_rows = [(48,168),(49,168),(64,392),(128,2048),(256,10240),(257,10496),
             (512,86016),(1024,532480),(4096,14712832),(8192,71368704),
             (32768,1543766016),(65536,7547387904),(524288,817966874624),
             (1048576,3718523125760)]
data = {row["N"]: row for row in json.loads((root/"outputs/finite-stages.json").read_text())}


def bracket_count(length, group, alpha):
    if not length:
        return 0
    incoming_switches = (2*length - 1 + alpha) // (2*alpha)
    outgoing_switches = (2*length - 2) // alpha
    pivots = length - 2*(group-1)*(incoming_switches+1) - 1
    counts = [(group, 0), (2*group-2, incoming_switches), (group-1, outgoing_switches)]
    return sum(copies*max(pivots-switches, 0) for copies, switches in counts)


print("N  r  p  d  X  degree  exhaustive-fold  max(E,S,F)")
for n, expected_fold in fold_rows:
    depth = max(1, max(i for i in range(n.bit_length()) if 2**i <= n)-7)
    cap = 2
    while cap < depth:
        cap *= 2
    assert data[n]["fold"]["q_max"] == cap
    horizontal = max(range(0, min(n//2, n-18*2**depth)+1, 2*cap))
    transverse = n-horizontal
    p = max(12*2**depth, transverse//2)
    d = transverse-p
    ends = [0]
    while ends[-1]+len(ends)+3 <= depth:
        ends.append(ends[-1]+len(ends)+3)
    total = 0
    for k in range(depth):
        j = depth-k
        e = 2**max(1, sum(end <= j for end in ends[1:]))
        q = min(cap, max(2, 2**((e.bit_length()-1)//2)))
        # Integer branch counts: each completed block replaces 2^length branches
        # by 2^length-4. All uncompleted ordinary factors are retained.
        lengths = [ends[i]-ends[i-1] for i in range(1,len(ends)) if ends[i-1]>=j]
        rows = 2**(k+1-sum(lengths)) * prod(2**length-4 for length in lengths)
        modes = p//2**(k+1) + (d+2**k)//2**(k+1)
        admissible = max(0, modes-2*e)
        best = max(bracket_count(length,q,2*e) for length in range(admissible+1))
        contribution = rows*(horizontal//q)*best
        assert contribution == data[n]["fold"]["stages"][k]["count"]
        total += contribution
    degree = max(2*p-1, 2*horizontal-1, 2*d)
    split = max(((2*i-1)*(2*(n-2-i)-3), -max(2*i-1, 2*(n-2-i)))
                for i in range(2,n-5))
    station, degree_station = split[0], -split[1]
    uniform = max(total, (n-2)**2//4, station)
    assert (degree,total) == (data[n]["fold"]["D"],expected_fold)
    assert (station, degree_station) == (data[n]["stationary"]["cycles"],data[n]["stationary"]["D"])
    assert uniform == data[n]["uniform"]["cycles"] and horizontal+p+d == n and degree <= n-1
    print(n, depth, p, d, horizontal, degree, total, uniform)

# Enumerate every admissible (depth, initial allowance) at the prescribed
# horizontal multiplicity. Iterate the bracket recurrence from zero.
for n in [32] + [n for n, _ in fold_rows]:
    k = n // 2
    cap = n - 1
    best_candidate = None
    for depth in range(1, n.bit_length()):
        for initial in range(4, (cap + 2) // (2**(depth+1)) + 1):
            powers = 2**depth
            size = powers * initial
            slots = 2*powers + size - 1 + k
            degree = max(4*powers-1, 2*size-2, 2*k-1)
            if slots > n or degree > cap:
                continue
            count = 0
            for step in range(depth):
                count = 2*count + (2**step)*initial - 1
            assert n <= 2*cap + k and count == depth*size//2-powers+1
            candidate = (2*k*count, depth, initial, slots, degree)
            if best_candidate is None or candidate > best_candidate:
                best_candidate = candidate
    witness = data[n]["separate_strips"]
    assert best_candidate == tuple(witness[key] for key in
                                    ("cycles","p","b","term_ceiling","degree_ceiling"))
    # At both parities, the pullback of the linear restoring force has
    # u and u^3; normalization removes the constant of F(phi).
    seed = data[n]["classical_seed"]
    degree = max(i for i in range(n) if 2*i <= cap)
    strips = n - degree - 2
    v_degrees = range(1, 2*strips, 2)
    u_degrees = range(2, 2*degree+1, 2)
    assert v_degrees[0] > 0 and u_degrees[0] > 0
    first_positions = len(v_degrees) + len(u_degrees)
    assert first_positions + 2 == n and seed["support"] == [first_positions, 2]
    assert seed["D"] == max(v_degrees[-1], u_degrees[-1], 3) <= cap
    half_count = 7*degree-48
    seed_count = half_count//2 + half_count%2
    assert seed["cycles"] == 2*strips*seed_count and degree >= 14
    print(f"N={n}: separate strips {witness['cycles']} at depth {witness['p']}, "
          f"b={witness['b']}, k={k}; linear restoring force {2*strips*seed_count} "
          f"from degree {degree} and {strips} strips.")
for n, expected in ((1024, (6,6,512,1089,1115136)), (8192, (8,14,4096,14081,115351552))):
    witness = data[n]["separate_strips"]
    assert tuple(witness[key] for key in ("p","b","k","per_sheet","cycles")) == expected
    assert witness["term_ceiling"] == witness["degree_ceiling"] == n-1
    assert witness["cycles"] > data[n]["uniform"]["cycles"]

for n, depth, initial, k, expected, degree in (
        (1024,6,7,449,1150338,897), (8192,8,15,3841,116036610,7681)):
    size = initial
    count = 0
    for _ in range(depth):
        count = 2*count + size - 1
        size *= 2
    assert 2*k*count == expected
    assert 2**(depth+1) + size - 1 + k == n
    assert max(2**(depth+2)-1, 2*size-2, 2*k-1) == degree < n
    print(f"N={n}: allowing k={k} gives {expected} by the same strip recurrence.")

for terms in range(9,1025):
    m = terms-2
    gs = max(2*(terms-r-4)*(r+1)+(terms-r-4)*(1+(-1)**r) for r in range(terms-3))
    assert gs == 4*(m**2//8)
    pairs = [(i,terms-2-i) for i in range(2,terms-5)]
    opt = max((2*i-1)*(2*j-3) for i,j in pairs if max(2*i-1,2*j)<terms)
    assert opt == (terms-4)**2-int(terms%2==0)
assert (data[32]["uniform"]["cycles"], data[32]["uniform"]["D"]) == (783,30)
print("N=9..1024: exhaustive original GS choices and feasible reduced stationary splits match both integer formulas.")
print("N=32: reduced-degree existence 783 at D=30; the fixed S4 field has D=31.")
# Substitutions in U3.1, S3.1, B2.1 and B1.1. All counts below are
# arithmetic consequences of the cited proofs.
for p, k, expected in ((2, 1, 32), (3, 4, 192)):
    count = 8*k*((9*p-2)//4)
    assert count == expected
    print(f"U3.1 p={p} k={k}: support {(p+2, k+1)}, lower bound {count}")

for n, d, added, expected in ((128, 127, (2, 2), 64005), (130, 129, (0, 0), 66045)):
    support = (2+added[0], n+d+added[1])
    degree = max(2*n-1, 2*d+1)
    count = (2*n-1)*(2*d-3)
    assert sum(support) == 261 and count == expected
    print(f"S3.1 n={n} d={d}: filled positions {added}, support {support}, degree {degree}, lower bound {count}")

M, d = 128, 7
base_count = M*(4*M-8*d+9)-12*(M-2*d+2)
assert M == 2**d and (base_count, base_count+9*M-40) == (58128, 59240)
print(f"B1.1 M={M} d={d}: support {(4, 2*M+1)}, degree {2*M-1}, F_*={base_count}, lower bound {base_count+9*M-40}")

# Read the printed cells independently of the table generator. The qualified
# degrees are algebraic expressions, so compare their TeX spelling as well.
small = {r["N"]: r for r in json.loads((root/"outputs/low-N.json").read_text())["low"]}
def table_cells(name):
    text = (root/"tex"/name).read_text()
    return {int(line.split(" & ")[0]):line.split(" & ")[1:]
            for line in text.splitlines() if line and line[0].isdigit()}

main_rows = table_cells("contribution-rows.tex")
supp_rows = table_cells("low-rows.tex")
outside = {4:[(4,7),(8,"S_L")], 5:[(8,7)], 6:[(32,r"E_6^\circ")],
           7:[(68,r"D_*^\circ")], 8:[(108,r"D_*^\circ")],
           9:[(148,r"D_*^\circ")], 10:[(192,r"E_{10}^\circ")]}
article_counts = [(1,3),(2,3),(4,5),(6,5),(9,7),(25,8),(35,8)]
extra_counts = {5:(4,4),6:(6,3),7:(10,5),8:(16,7)}
published_capped = {4:"1 (3)", 5:"3 (3)", 6:"$4^{a}$ (3)", 7:"$5^{a}$ (3)", 8:"5 (6)", 9:"24 (4)", 10:"$24^{p}$ (4)"}
for n,(c,d) in enumerate(article_counts, start=4):
    row = main_rows[n]
    extra = [str(x) for x in extra_counts[n]] if n in extra_counts else ["---", "---"]
    assert row[:-1] == [str((n-2)**2//4), str(c), str(d), *extra]
    assert row[-1].removesuffix(r" \\") == published_capped[n]
    assert small[n]["uniform"] == [c,d,"H","+"]
    if n in extra_counts:
        c,d = extra_counts[n]
    for count,degree,cap in [(c,d,"+"),*((k,v,"-") for k,v in outside[n])]:
        assert [count,degree,"H",cap] in small[n]["specialized"]
        star = r"^{\star}" if n in (4,5,6) and (count,degree)==(4*(n-3),7 if n<6 else 9) else ""
        assert "$" + rf"{count}{star},\,{degree},\,H,\,{cap}$" in supp_rows[n][-1]
for n in (4,5,6):
    assert [8,"S_L","H","-"] in small[n]["specialized"]
    assert r"$8,\,S_L,\,H,\,-$" in supp_rows[n][-1]
for record in ((6,4,"H","+"),(12,9,"H","-")):
    assert list(record) in small[6]["specialized"]
    marker = r"^{\star}" if record[0]==12 else ""
    assert "$"+str(record[0])+marker+r",\,"+r",\,".join(map(str,record[1:]))+"$" in supp_rows[6][-1]
for n,count,degree,kind in ((4,4,7,"H"),(5,8,7,"I"),(6,12,9,"I")):
    assert [count,degree,kind,"-"] in small[n]["published"]
    flag = "^{f}" if n==4 else ""
    assert "$" + rf"{count}{flag},\,{degree},\,{kind},\,-$" in supp_rows[n][0]
    assert "$" + rf"{count}^{{\star}},\,{degree},\,H,\,-$" in supp_rows[n][-1]
assert {(24,"G_3","O","?"),(24,4,"I","+")} <= set(map(tuple,small[9]["published"]))
assert (32,"G_4","O","?") in map(tuple,small[10]["published"])
print("Tables 3 and U1: seven article counts and degrees, published capped bounds, four additional capped fields in both tables and three attributed GS double lifts checked.")
print("Detailed U1 entries include the three power-law lifts, both six-term focus lifts, and distinct GS/BBJT witnesses.")
