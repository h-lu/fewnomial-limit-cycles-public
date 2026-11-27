"""Exercise the half-shifted grid count and protected pivot of Lemma B1.5."""
from fractions import Fraction as F
from math import ceil, floor


def check_window(alpha, p, a, b):
    pivot = alpha * (2*p - 1) // 2
    candidates = [j for j in range(pivot, alpha*p) if a < F(2*j+1, 2*alpha) < b]
    exact_count = ceil(alpha*b - F(1, 2)) - floor(alpha*a - F(1, 2)) - 1
    assert len(candidates) == exact_count >= alpha*(b-a) - 1
    retained = [j for j in candidates if j > pivot]
    assert len(retained) >= alpha*(b-a) - 2
    for h in range(1, 7):
        if alpha*(b-a) > h*(p+1) + 3:
            used = retained[:h*(p+1)]
            assert len(used) == h*(p+1)
            assert used == list(range(used[0], used[0]+len(used)))
            assert all(p < F(2*j+1, 2*alpha) + F(1, 2) < p+1 for j in used)


if __name__ == "__main__":
    assert F(2*64+1, 256) == F(129, 256) > F(1, 2)
    cases = 0
    for alpha in (4, 6, 8, 16, 32, 64, 128, 256):
        for p in (1, 2, 3, 4):
            low, high = F(2*p-1, 2), F(p)
            # Endpoints on a node, on its half-cell, and strictly between them.
            endpoints = sorted({low+F(i, 4*alpha) for i in (0, 1, 2, 3, alpha)}
                               | {high-F(i, 4*alpha) for i in (0, 1, 2, 3, alpha)})
            for i, a in enumerate(endpoints):
                for b in endpoints[i+1:]:
                    check_window(alpha, p, a, b)
                    cases += 1
    print("alpha=128, p=1, j=64 gives g=129/256, with j equal to protected pivot 64.")
    print(f"{cases} exact open-window counts; all admissible test blocks exclude the protected prefix.")
    print("Finite M=128 uses columns 65..74 in five consecutive pairs; determinants replay in verify_paper.py.")
