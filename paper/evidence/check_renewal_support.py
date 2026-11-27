"""Check ordinary coefficients of the published two-source renewal over Q."""
from argparse import ArgumentParser
from hashlib import md5, sha256
import json
from math import comb
from pathlib import Path

from sympy import QQ
from sympy.polys.rings import ring

ROOT = Path(__file__).resolve().parents[1]
R, u, v = ring("u,v", QQ)


def primary_record(downloads=None):
    directory = ROOT / "primary" / "zenodo-23113313"
    record = json.loads((directory / "record.json").read_text())
    assert record["doi"] == "10.5281/zenodo.23113313"
    assert record["conceptrecid"] == "21891383"
    assert record["metadata"]["version"] == "submission-2026-10-03"
    assert record["metadata"]["license"]["id"] == "cc-by-4.0"
    files = {item["key"]: item for item in record["files"]}
    identities = {
        "manuscript.pdf": (452637, "cab50930dee9a3b858c266c83570a3cf",
                           "c65be693ff8643923c9279a01b56604e86b4409935fd5d158981951a6bb63254"),
        "manuscript.tex": (89365, "3c408db815a40e8abb40194fbff5d69c",
                           "803c75f9a36c820c49e620443da6e5a11fbf1b5d89a051d717b065c4155f33f7"),
    }
    for name, (size, digest, sha) in identities.items():
        assert files[name]["checksum"] == "md5:" + digest and files[name]["size"] == size
        print(f"Zenodo record: {name}, {size} bytes, MD5 {digest}")
        if downloads is not None:
            data = (downloads / name).read_bytes()
            assert len(data) == size and md5(data).hexdigest() == digest
            assert sha256(data).hexdigest() == sha
            print(f"Supplied primary bytes verified: {name}, SHA-256 {sha}")


def common_odd(t, b, delta):
    return sum((-1)**k * delta**((b-1-k)*(b-k)//2) * t**(2*k+1) for k in range(b))


def translated_coefficients(poly, tau):
    # Coefficient of U^i V^j after the two affine substitutions of supplement S1.1.
    coefficients = {}
    for (a, b), c in poly.items():
        scale = c / tau**(a+b)
        for i in range(a + 1):
            for j in range(b + 1):
                key = (i, j)
                term = (-1)**(a+b-i-j) * comb(a, i) * comb(b, j) * scale
                coefficients[key] = coefficients.get(key, QQ.zero) + term
    return {ij: c for ij, c in coefficients.items() if c}


def direct_step(seed, m, tau):
    b = m + 2
    p = R.one
    for i in range(b):
        p *= u - QQ(2*i-b+1, 2*b)
    dp = p.diff(u)
    corner = [(u, -R.one)]
    lam = p.compose(corner)[(0, 0)] / dp.compose(corner)[(0, 0)]
    assert lam < 0
    swap = [(u, v)]
    sx = p.compose(swap) + lam * (dp - dp.compose(corner))
    sy = p + lam * (dp.compose(swap) - dp.compose(corner))
    mask = (1 + u)*(1 + v)
    scaled = [(u, u / tau), (v, v / tau)]
    zx, zy = [s + mask * a.compose(scaled) / tau for s, a in zip((sx, sy), seed)]
    assert zx.compose(corner) == p.compose(swap)
    assert zy.compose([(v, -R.one)]) == p
    assert zy.compose(corner) == lam * dp.compose(swap)
    assert zx.compose([(v, -R.one)]) == lam * dp

    fold = [(u, 2*u*u-1), (v, 2*v*v-1)]
    lift = 4*v*zx.compose(fold), 4*u*zy.compose(fold)
    assert 4*u*lift[0] == 16*u*v*zx.compose(fold)
    assert 4*v*lift[1] == 16*u*v*zy.compose(fold)
    # Rational algebraic probes. No dynamical threshold is assumed for these values.
    odd_u, odd_v = common_odd(u, b, QQ(1, 2)), common_odd(v, b, QQ(1, 3))
    out = (lift[0] + QQ(1, 11)*v*odd_u, lift[1] + QQ(1, 13)*u*odd_v)
    sizes = []
    for component, inherited, odd in zip(range(2), seed, (odd_u, odd_v)):
        translated = translated_coefficients(inherited, tau)
        # The edge and perturbation slots are disjoint from the translated interior.
        edge = {(0, 2*j+1) for j in range(b+1)} | {(2*i, 1) for i in range(1, b)}
        perturb = {(2*k+1, 1) for k in range(b)}
        if component == 1:
            edge = {(j, i) for i, j in edge}
            perturb = {(j, i) for i, j in perturb}
        i0, j0 = (2, 3) if component == 0 else (3, 2)
        inside = {(2*i+i0, 2*j+j0) for i, j in translated}
        assert edge.isdisjoint(inside | perturb) and perturb.isdisjoint(inside)
        assert len(odd) == b and len(edge) == 2*b
        assert set(out[component]) == edge | inside | perturb
        for (i, j), coefficient in translated.items():
            assert out[component][(2*i+i0, 2*j+j0)] == 16 * 2**(i+j) * coefficient / tau
        assert max(i+j for i, j in out[component]) == 2*b+1
        sizes.append(len(translated))
    assert sum(map(len, out)) == 6*b + sum(sizes)
    return out, sizes


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--primary-files", type=Path, help="also verify separately obtained PDF and TeX bytes")
    primary_record(parser.parse_args().primary_files)
    empty = (R.zero, R.zero)
    for m in (3, 4, 5, 7, 10, 13):
        out, sizes = direct_step(empty, m, QQ(1, 5))
        b = m + 2
        assert sizes == [0, 0] and [len(p) for p in out] == [3*b, 3*b]
        assert out[0].gcd(out[1]) == R.one
        target_terms = {5: 33, 7: 47, 15: 90}.get(b)
        if target_terms is not None:
            assert 6*b <= target_terms <= 32*33 and 2*b+1 <= 31
            assert 4*b*(b-1) == {33: 80, 47: 168, 90: 840}[target_terms]
            print(f"D <= 31, fill {6*b} to {target_terms}: {4*b*(b-1)} hyperbolic cycles by Proposition 4.1.")
        print(f"zero seed, m={m}: component terms {3*b}+{3*b}, D={2*b+1}, "
              f"analytic selected count 4b(b-1)={4*b*(b-1)}")

    # Opposite coefficients of the same homogeneous degree can cancel for all tau.
    seed = (u-v, u+1)
    for tau, expected in ((QQ.one, [2, 1]), (QQ(1, 7), [2, 2])):
        out, sizes = direct_step(seed, 3, tau)
        assert sizes == expected
        print(f"cancellation probe tau={tau}: inherited slots {sizes}, output terms {sum(map(len, out))}")

    seed = empty
    m = 3
    for j in range(1, 4):
        seed, sizes = direct_step(seed, m, QQ(1, 17))
        m = 2*m+5
        a, b = 2**j-2, 7*2**j-3
        for field, corner in zip(seed, ((a, b), (b, a))):
            top = {ij: c for ij, c in field.items() if sum(ij) == m}
            assert corner in top and all(c > 0 for c in top.values())
        if j == 2:
            assert sizes == [30, 30] and sum(map(len, seed)) == 138
            assert seed[0].gcd(seed[1]) == R.one
        print(f"dyadic algebra j={j}: D={m}, inherited slots {sizes}, "
              f"component terms {[len(p) for p in seed]}, top exponents {(a,b)} and {(b,a)}")
    print("Component gcd is one in the six zero-seed probes and the degree-27 probe.")
    # The printed n=64 example uses two odd binary remainders.
    L = 0
    counts = []
    for m, target in ((3, 12), (12, 29), (29, 64)):
        assert target - (2*m+5) in (0, 1)
        L = 4*L + 4*(m+2)*(m+1)
        counts.append(L)
    assert counts == [80, 1048, 7912]
    print("n=64 binary recurrence: 80, 1048, 7912. Source (25)-(26).")
    print("Exact Q coefficient identities and occupied slots. Cycle counts use the analytic renewal theorem.")


if __name__ == "__main__":
    main()
