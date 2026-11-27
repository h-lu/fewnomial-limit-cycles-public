"""Integer allocation for the finite lower bound. No continuous search."""
import argparse
import json
from fractions import Fraction


def endpoint_count(q, alpha, length):
    if length == 0:
        return 0
    b = (length - 1 + alpha // 2) // alpha
    c = 2 * (length - 1) // alpha
    j = length - 2 * (q - 1) * (b + 1) - 1
    return q * max(0, j) + 2 * (q - 1) * max(0, j - b) + (q - 1) * max(0, j - c)


def folded(terms):
    if terms < 48:
        return None
    r = max(1, terms.bit_length() - 8)
    power = 1 << r
    group = 1 << (max(2, r) - 1).bit_length()
    x = 2 * group * (min(terms // 2, terms - 18 * power) // (2 * group))
    t = terms - x
    p = max(12 * power, t // 2)
    d = t - p
    degree = max(2 * p - 1, 2 * x - 1, 2 * d)
    assert x >= 2 * group and p >= 12 * power and d >= 6 * power
    assert (x // group) % 2 == 0 and p + d + x == terms and degree < terms
    ends = [0]
    i = 1
    while ends[-1] + i + 3 <= r:
        ends.append(ends[-1] + i + 3)
        i += 1
    stages = []
    for k in range(r):
        j = r - k
        order = 1 << max(1, sum(b <= j for b in ends[1:]))
        q = min(group, 1 << max(1, (order.bit_length() - 1) // 2))
        alpha = 2 * order
        y = Fraction(1 << (k + 1))
        for i in range(1, len(ends)):
            if r - ends[i - 1] <= k:
                y *= 1 - Fraction(4, 1 << (i + 3))
        assert y.denominator == 1
        modes = (p >> (k + 1)) + ((d + (1 << k)) >> (k + 1))
        length = max(0, modes - alpha)
        # A full period cannot lower any of the three positive parts.
        # Choose the longest prefix in case of a tie.
        b, used = max((endpoint_count(q, alpha, l), l)
                      for l in range(max(0, length - alpha + 1), length + 1))
        stages.append(dict(k=k, j=j, Y=int(y), e=order, q=q, alpha=alpha,
                           M=modes, L=length, used=used, B=b, count=int(y) * (x // q) * b))
    return dict(r=r, R=power, q_max=group, s=x // group, X=x, p=p, d=d,
                support=[p, x + d], D=degree, cycles=sum(v["count"] for v in stages),
                stages=stages)


def stationary(terms):
    """Optimized two-sided stationary-turn construction after density reduction."""
    if terms < 9:
        return None
    n = (terms - 2) // 2
    d = terms - 2 - n
    degree = max(2 * n - 1, 2 * d)
    return dict(method="stationary", n=n, d=d, support=[2, n + d],
                D=degree, cycles=(2 * n - 1) * (2 * d - 3))


def gs_proposition(terms):
    """Maximum of Gasull--Santana Proposition 1 at the displayed budgets."""
    if terms < 9:
        raise ValueError("The comparison is used for N >= 9")
    return 4 * ((terms - 2)**2 // 8)


def separate_strips(terms):
    """Largest terminal-fold count at the prescribed k=floor(N/2)."""
    k = terms // 2
    cap = terms - 1
    choices = []
    for p in range(1, terms.bit_length()):
        power = 1 << p
        if 4 * power - 1 > cap:
            break
        max_s = min((cap + 2) // 2, terms - k - 2 * power + 1)
        b = max_s // power
        if b < 4:
            continue
        s = b * power
        per_sheet = p * s // 2 - power + 1
        choices.append(dict(p=p, b=b, k=k, s=s, per_sheet=per_sheet,
                            term_ceiling=k + 2 * power + s - 1,
                            degree_ceiling=max(2*k - 1, 4*power - 1, 2*s - 2),
                            cycles=2 * k * per_sheet))
    return max(choices, key=lambda row: (row["cycles"], row["p"]), default=None)


def classical_seed(terms):
    """Quadratic pullback of the November linear-restoring-force seed."""
    n = (terms - 1) // 2
    if n < 14:
        return None
    k = terms - n - 2
    seed_cycles = (7*n - 47) // 2
    return dict(n=n, k=k, seed_cycles=seed_cycles, support=[k + n, 2],
                D=max(2*k - 1, 2*n, 3), cycles=2*k*seed_cycles)


def result(terms):
    if terms < 1:
        raise ValueError("N must be a positive integer")
    if terms < 4:
        supports = ((1, 0), (1, 1), (1, 2))
        base = dict(method="zero", support=supports[terms-1], D=int(terms==3), cycles=0)
    else:
        n = (terms - 1) // 2
        t = terms - 1 - n
        base = dict(method="base", n=n, t=t, support=[1, n + t],
                    D=2 * max(n, t) - 1, cycles=n * (t - 1))
    f = folded(terms)
    turn = stationary(terms)
    choices = [base]
    if turn:
        choices.append(turn)
    if f:
        choices.append(dict(method="fold", support=f["support"], D=f["D"], cycles=f["cycles"]))
    winner = max(choices, key=lambda v: v["cycles"])
    # "uniform" is retained for callers selecting only the original E/S/F rules.
    return dict(N=terms, base=base, stationary=turn, fold=f, uniform=winner,
                separate_strips=separate_strips(terms), classical_seed=classical_seed(terms))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, nargs="+")
    parser.add_argument("--stages", action="store_true", help="include every stage contribution")
    args = parser.parse_args()
    for n in args.N:
        row = result(n)
        if row["fold"] and not args.stages:
            del row["fold"]["stages"]
        print(json.dumps(row, ensure_ascii=True))

