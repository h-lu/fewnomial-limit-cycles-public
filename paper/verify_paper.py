#!/usr/bin/env python3
"""
Finite checks for "Hyperbolic limit cycles with few monomials under a linear degree bound".

Requires Python 3 and `python-flint` (for exact rational polynomials and
outward-rounded Arb ball arithmetic). Zero external project dependencies.
"""
from __future__ import annotations

from fractions import Fraction
from math import prod
from pathlib import Path
from typing import List, Tuple

from flint import acb, acb_poly, arb, arb_mat, ctx, fmpq as Q, fmpq_poly as Poly


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"Verification failed: {msg}")


# ------------------------------------------------------------------------------
# 1. Recurrent quadratic folds, whole-disk coprime division, and joint budgets
# ------------------------------------------------------------------------------
def fold_schedule(r: int) -> Tuple[List[Tuple[int, int, int]], int, Fraction]:
    b_ends = [0]
    i = 1
    while b_ends[-1] + (i + 3) <= r:
        b_ends.append(b_ends[-1] + (i + 3))
        i += 1
    I_max = len(b_ends) - 1
    rows: List[Tuple[int, int, int]] = []
    loss = Fraction(0)
    e_inv = Fraction(0)
    deg_sum = 0
    for k in range(r):
        j = r - k
        i0 = sum(1 for m in range(1, I_max + 1) if b_ends[m] <= j)
        e_j = 2 ** max(1, i0)
        yk_frac = Fraction(2 ** (k + 1), 1)
        for m in range(1, I_max + 1):
            if r - b_ends[m - 1] <= k:
                yk_frac *= Fraction(1, 1) - Fraction(4, 2 ** (m + 3))
        check(yk_frac.denominator == 1, "Y_k integrality")
        yk = int(yk_frac)
        rows.append((k, yk, e_j))
        loss += Fraction(1, 1) - Fraction(yk, 2 ** (k + 1))
        e_inv += Fraction(1, e_j)
        deg_sum += (2 ** (k + 1)) * e_j
    check(loss <= Fraction(5, 2) and e_inv <= 10 and deg_sum < 6 * (2 ** r), "Uniform fold bounds")
    return rows, deg_sum, loss


def verify_flatness(r: int) -> None:
    """Truncated z-jets over Q[t]/Phi_(2D), t a primitive (2D)-th root."""
    rows, _, _ = fold_schedule(r)
    blocks = {}
    end = 0
    length = 4
    while end + length <= r:
        end += length
        blocks[r - end] = length
        length += 1
    largest_length = max(blocks.values())
    degree = 1 << largest_length
    t = Poly([0, 1])
    modulus = t**degree + 1
    one = Poly([1])
    # t^(-s) = -t^(D-s) in this cyclotomic field.
    tuning = {}
    for stage, length in blocks.items():
        power = 1 << (largest_length - length)
        tuning[stage] = (t**power - t**(degree - power)) / 2

    for k, _, order in rows:
        jet = [Poly([]) for _ in range(order + 1)]
        jet[1] = one
        for stage in range(k, r):
            scale = tuning.get(stage, one)
            square = [sum((jet[i] * jet[j - i] for i in range(j + 1)), Poly([]))
                      for j in range(order + 1)]
            jet = [(2 * coef * scale) % modulus for coef in square]
            jet[0] = (jet[0] - scale) % modulus
        check(jet[0] in (-one, Poly([]), one), "Exact critical image")
        check(all(not coef for coef in jet[1:order]), "Exact zero coefficients below the order")
        check(bool(jet[order]), "Exact nonzero leading coefficient")


def verify_recurrent_and_joint() -> None:
    for r in range(1, 40):
        fold_schedule(r)
    for r in (4, 7, 9):
        verify_flatness(r)
    print("Exact cyclotomic jets at depths 4, 7, 9 (20 stage orders).")

    # Exact rational Bezout division with local factor cancellation C_V(z^2) | H
    Z = Poly([0, 1])
    for p_next, d_next, d_cur, p_cur, e_ord in ((12, 6, 13, 25, 2), (17, 9, 19, 35, 4)):
        Vp = Poly([Q((-1) ** i, i + 2) for i in range(p_next - 1)] + [Q(1)])
        Fp = Poly([Q(i + 1, i + 3) for i in range(d_next - 1)] + [Q(1)])
        check(Vp.gcd(Fp) == Poly([1]), "Coprime stage derivatives")
        J = 2 * Z * Z - 1
        D0 = Vp(J)
        E0 = Fp(J)
        C_loc = (Z * Z + Q(1, 7)) ** (e_ord - 1)
        D0_mod = D0 * C_loc
        p_next_eff = p_next + e_ord - 1
        p_cur_eff = 2 * p_next_eff + 1
        check(D0_mod.gcd(E0) == Poly([1]), "Coprime composed derivatives")
        N_hat = Poly([Q((i % 5) + 1, 3) for i in range(2 * p_next_eff - 2 + d_cur - C_loc.degree() + 1)])
        N0 = C_loc * N_hat
        g, _, inv_e = D0_mod.xgcd(E0)
        check(g == Poly([1]), "Extended GCD unit")
        H = (-N0 * inv_e) % D0_mod
        B, rem = divmod(N0 + E0 * H, D0_mod)
        check(rem == Poly([]) and D0_mod * B - E0 * H == N0, "Exact Bezout identity")
        check(H.degree() < p_cur_eff and B.degree() <= d_cur, "Degree caps on H and B")
        check(H % C_loc == Poly([]), "Local root factor C_loc divides H")

    # Five illustrative allocations for Theorem 5.1.
    for m, Delta in ((600, 600), (601, 600), (900, 899), (5000, 600), (900, 1500)):
        h = min(m, Delta)
        r = (h // 128).bit_length() - 1
        Q_max = 1 << (max(2, r) - 1).bit_length()
        H_st = h // 2
        t_st = min(2 * H_st, m - H_st)
        p = t_st // 2
        d = t_st - p
        s = 2 * (H_st // (2 * Q_max))
        X_st = s * Q_max
        m0 = p + d + X_st
        B_del = (Delta + 1) * (Delta + 2) // 2
        check(m0 <= m <= 2 * B_del and max(2 * p - 1, 2 * X_st - 1, 2 * d) <= Delta, "Natural budget")
        i_p = min(m - m0, B_del - p)
        i_q = (m - m0) - i_p
        check(0 <= i_p <= B_del - p and 0 <= i_q <= B_del - (X_st + d), "Padding slots")
        if Delta in (m - 1, m):
            d_unpad = m - p - X_st
            check(d_unpad >= d and max(2 * p - 1, 2 * X_st - 1, 2 * d_unpad) <= Delta,
                  "Unpadded 3-row exact support at linear degree cap")

    # Summing the finite estimate: sum_m 2^-m = 1 and sum_m m*2^-m = 2, starting at m=1.
    reciprocal_group = 1 + 4 + 4 * 2 + 9
    group_over_order = 8 + 2 * 2 + 4 + Fraction(2 * 2 + 5, 2)
    check((reciprocal_group, group_over_order) == (22, Fraction(41, 2)), "Grouped sums")
    loss_terms = (
        Fraction(7),                         # b-r
        Fraction(5, 2),                      # branch loss
        Fraction(3 * reciprocal_group, 4),   # 3 L/q
        group_over_order,                    # 8 q L/alpha
        Fraction(24 + 36, 128),              # flatness and 12 q
    )
    coefficient = sum(loss_terms)
    check(coefficient == Fraction(1503, 32) and coefficient < 50, "Current dyadic loss")
    print("Dyadic loss terms:", " + ".join(map(str, loss_terms)), "=", coefficient, "< 50")
    print("Five sample joint allocations and their available padding slots passed.")
    print("[1/3] Recurrent folds, coprime division and joint budgets: OK")


# ------------------------------------------------------------------------------
# 2. Stationary-turn 64,005 baseline & explicit 783-cycle rational witness
# ------------------------------------------------------------------------------
def F_native(h: int, L: int, alpha: int) -> int:
    if L <= 0:
        return 0
    N = 2 ** h
    b = (L - 1 + alpha // 2) // alpha
    c = (2 * (L - 1)) // alpha
    J = L - h * (b + 1) - 1
    return 0 if J < 0 else N * J + (N - 1) * (2 * max(J - b, 0) + max(J - c, 0))


def unres_opt(h: int, alpha: int, l: int) -> int:
    if l <= 0:
        return 0
    b = (l - 1 + alpha // 2) // alpha
    if l - h * (b + 1) - 1 < 0:
        return 0
    N = 2 ** h
    term = 2 * (N - 1) if (h <= alpha // 2 and h <= l) else 0
    bridge = (2 * N - 2) if (2 * l <= alpha) else ((N - 2) if l <= alpha else 0)
    return F_native(h, l, alpha) + term + bridge


def verify_stationary_and_witness() -> None:
    # Check exponent identities and nonvanishing highest-q-power control coefficients
    for d_chk in range(4, 150):
        r_chk = 2 * d_chk - 1
        b_chk = d_chk - 2
        e_b = r_chk + 4 * b_chk + 2
        max_lower = r_chk + 4 * (b_chk - 1) + 2
        check(e_b == 3 * r_chk - 4 and max_lower == 3 * r_chk - 8, "e_b=3r-4 and max_{j<b}e_j=3r-8")
        check(e_b > 2 * r_chk and max_lower < min(e_b, 4 * r_chk), "Fractional/smooth separation")

    # Same-budget comparison at (4,257), D=255 (m=261)
    gs_261 = max(2 * n_g * (r_g + 1) + n_g * (1 + (-1) ** r_g)
                 for r_g in range(257) for n_g in [257 - r_g])
    raw_128 = 4 * F_native(5, 128, 128)
    unres_128 = 4 * unres_opt(5, 128, 128)
    inter_128 = raw_128 + 9 * 128 - 40
    st_261 = (2 * 128 - 1) * (2 * 127 - 3)
    check((gs_261, raw_128, unres_128, inter_128, st_261) == (33540, 58128, 58496, 59240, 64005),
          "(4,257), D=255 benchmark tuple")

    # Explicit polynomial support check for Corollary B2.1 at n=128, d=127 + 4-slot pad
    Z = Poly([0, 1])
    t0_c, t1_c = Poly([1]), Z
    for _ in range(1, 128):
        t0_c, t1_c = t1_c, 2 * Z * t1_c - t0_c
    u_prime_128 = -(t1_c ** 2).derivative()
    p_unpad_supp = {(0, 252), (0, 254)}
    q_unpad_supp = {(j, 0) for j, v in enumerate(u_prime_128) if v} | (
        {(0, 2 * j + 1) for j in range(126)} | {(0, 255)}
    )
    check((len(p_unpad_supp), len(q_unpad_supp)) == (2, 255), "Unpadded (2,255) support")
    p_pad_supp = p_unpad_supp | {(0, 0), (0, 1)}
    q_pad_supp = q_unpad_supp | {(0, 0), (2, 0)}
    max_deg_pad = max(i + j for i, j in (p_pad_supp | q_pad_supp))
    check((len(p_pad_supp), len(q_pad_supp), max_deg_pad) == (4, 257, 255),
          "Exact padded (4,257), D=255 support")

    # Explicit 783-cycle rational witness at (2,30), D=31
    with ctx.workprec(1000):
        n, d, r, B_exp = 15, 15, 29, 96
        b_idx = d - 2
        q_val = Q(1, 2) ** 64

        def beta_arb(u, v) -> arb:
            return arb(u).gamma() * arb(v).gamma() / arb(u + v).gamma()

        r0, r1 = [Q(1)], [Q(1)]
        for j in range(d):
            x, t = Q(2 * j + 1, 2) + Q(r, 4), Q(2 * j + 1, 2) - Q(r, 4)
            r0.append(r0[-1] * x / (x + Q(3, 2)))
            r1.append(r1[-1] * t / (t + Q(1, 2)))
        R0 = [v / r0[b_idx] for v in r0]
        R1 = [v / r1[b_idx] for v in r1]
        check(R0[d] > R1[d] > 0, "Control ratio inequality R_{0d} > R_{1d} > 0")

        def L_ctrl(s_v: Q, t_v: Q) -> Poly:
            cd = (s_v - t_v) / (R0[d] - R1[d])
            return (s_v - R0[d] * cd) * (Z ** (2 * b_idx)) + cd * (Z ** (2 * d))

        L10 = L_ctrl(Q(1), Q(0))
        check(L10[2 * b_idx] != 0 and L10[2 * d] != 0, "Both L(1,0) control coefficients nonzero")

        s0 = beta_arb(b_idx + Q(1, 2) + Q(r, 4), Q(3, 2)) / 2
        s1 = beta_arb(b_idx + Q(1, 2) - Q(r, 4), Q(1, 2)) / 4
        dirs = [(0, L10, s0, s0), (2 * r, L_ctrl(Q(0), Q(1)), s1, -s1)]
        for j in range(b_idx):
            al = Q(2 * j + 1, r)
            bj = beta_arb(al, Q(3, 2)) / r
            cp, cm = bj * (1 + 1 / arb(al).cos_pi()), -bj * arb(al).tan_pi()
            nu = al + Q(1, 2)
            direct_cp = (beta_arb(al, -nu) + beta_arb(al, Q(3, 2))) / r
            direct_cm = beta_arb(-nu, Q(3, 2)) / r
            check(cp.overlaps(direct_cp) and cm.overlaps(direct_cm) and cp * cm < 0,
                  "Reflection identity and opposite two-sided signs")
            dirs.append((r + 4 * j + 2, (Z ** (2 * j)) + L_ctrl(-R0[j], -R1[j]), cp, cm))
        dirs.sort(key=lambda it: it[0])
        check([it[0] for it in dirs] == [0, 31, 35, 39, 43, 47, 51, 55, 58, 59, 63, 67, 71, 75, 79],
              "Sorted leading exponents")

        G_poly = Poly([])
        for i, (ep, Dp, cp, cm) in enumerate(dirs):
            sgn_p = 1 if cp > 0 else -1
            G_poly += ((-1) ** i) * sgn_p * (q_val ** ((B_exp - ep) ** 2)) * Dp
            check(sum(abs(Dp[k]) for k in range(Dp.degree() + 1)) < 2 ** 40, "D_i l1 norm < 2^40")
            check(arb(2) ** -12 < abs(cp) < arb(2) ** 12 and arb(2) ** -12 < abs(cm) < arb(2) ** 12,
                  "2^-12 < |c_{i,+-}| < 2^12")
        F_poly = G_poly.integral()
        check(F_poly[0] == 0 and F_poly.derivative() == G_poly, "Exact polynomial primitive F")
        check(sum(abs(F_poly[k]) for k in range(F_poly.degree() + 1)) < 2 ** 60, "F l1 norm < 2^60")

        t0, t1 = Poly([1]), Z
        for _ in range(1, n):
            t0, t1 = t1, 2 * Z * t1 - t0
        horiz = -(t1 ** 2).derivative()
        p_terms = [(0, r - 1, Q(-r)), (0, r + 1, Q(r + 2))]
        q_x_terms = [(j, 0, v * (Q(2) ** 20_000_000)) for j, v in enumerate(horiz) if v]
        q_y_terms = [(0, j, v * (Q(2) ** -10_000_000)) for j, v in enumerate(F_poly) if v]
        check(len(p_terms) == 2 and len(q_x_terms) == 15 and len(q_y_terms) == 15,
              "Exact (2,30) collected monomial counts")
        check([ye for _, ye, _ in q_y_terms] == [2 * j + 1 for j in (*range(14), 15)],
              "All 15 drive odd powers survive")
        check(max(xe + ye for xe, ye, _ in p_terms + q_x_terms + q_y_terms) == 31, "Actual degree 31")

        # Certify all 27 explicit dyadic rational brackets in Arb ball arithmetic
        brackets = ["# w endpoints = mantissa * 2^shift; h = sign * w^58. "
                    "Last three columns bound the left value, right value and slope of the normalized model in units of 2^-20.",
                    "sign\te\tf\tleft_mantissa\tright_mantissa\tshift\tleft_lower\tright_upper\tslope_upper"]
        quadrature = ["# Each upper bound encloses |I[D]/leading - 1|, in units of 2^-36.",
                      "left_turn\texponent\terror_upper"]
        counts = []
        for side_idx in (0, 1):
            consts = [((-1) ** i) * (1 if cp > 0 else -1) * (cp if side_idx == 0 else cm)
                      for i, (_, _, cp, cm) in enumerate(dirs)]
            bands = []
            previous_right = Q(0)
            for i in range(len(dirs) - 1):
                if consts[i] * consts[i + 1] > 0:
                    continue
                e, f = dirs[i][0], dirs[i + 1][0]
                delta = f - e
                t0_val = abs(consts[i] / consts[i + 1]) ** (arb(1) / delta)
                logw = -64 * (2 * B_exp - e - f) + t0_val.log() / arb(2).log()
                check(logw > -10998 and logw < -2002, "w scale inside (2^-11000, 2^-2000)")
                if bands:
                    check(logw > bands[-1] + 2, "Pairwise disjoint brackets")
                bands.append(logw)
                step = arb(1) / (5 * delta)
                lo = ((1 - step) * t0_val * (2 ** 80)).floor().unique_fmpz()
                hi = ((1 + step) * t0_val * (2 ** 80)).ceil().unique_fmpz()
                check(lo is not None and hi is not None, "Unique outward integer rounding")
                vl, vr = arb(lo) / ((2 ** 80) * t0_val), arb(hi) / ((2 ** 80) * t0_val)
                check(vl > 1 - arb(1) / (4 * delta) and vr < 1 + arb(1) / (4 * delta),
                      "Rational subinterval inside |v-1| < 1/(4*Delta)")
                v_int = vl.union(vr)
                mults = [(dirs[k][0] - e, (consts[k] / consts[i]) * (t0_val ** (dirs[k][0] - e))
                          * (arb(2) ** (-64 * (dirs[k][0] - e) * (dirs[k][0] - f))))
                         for k in range(len(dirs))]
                left_v = sum(c_m * (vl ** pw) for pw, c_m in mults)
                right_v = sum(c_m * (vr ** pw) for pw, c_m in mults)
                slope_v = sum(pw * c_m * (v_int ** (pw - 1)) for pw, c_m in mults if pw)
                check(left_v > Q(1, 8) and right_v < Q(-1, 8) and slope_v < -arb(delta) / 2,
                      "Strict endpoint sign and derivative margins")
                shift = -80 - 64 * (2 * B_exp - e - f)
                w_left, w_right = Q(lo) * Q(2)**shift, Q(hi) * Q(2)**shift
                check(Q(2)**-10998 < w_left < w_right < Q(2)**-2002 and previous_right < w_left,
                      "Disjoint exact w endpoints inside the stated window")
                previous_right = w_right
                # Monotone w^e and the fixed coefficient bound the normalizing factor on the full bracket.
                scale_min = abs(consts[i]) * arb(q_val ** ((B_exp - e)**2)) * arb(w_left)**e
                check(scale_min > arb(2)**-2_000_000, "S4 normalizing factor on the closed bracket")
                lower = int((left_v.lower() * 2**20).floor().unique_fmpz())
                upper = int((right_v.upper() * 2**20).ceil().unique_fmpz())
                slope = int((slope_v.upper() * 2**20).ceil().unique_fmpz())
                check(lower > 2**17 and upper < -2**17 and slope < -delta * 2**19,
                      "Outward dyadic export retains all three strict margins")
                sign = 1 if side_idx == 0 else -1
                brackets.append(f"{sign}\t{e}\t{f}\t{lo}\t{hi}\t{shift}\t{lower}\t{upper}\t{slope}")
            counts.append(len(bands))
        check(counts == [14, 13] and (2 * n - 1) * sum(counts) == 783, "29 * (14 + 13) = 783")

        # Check analytic Rouche/Cauchy constants of Section S4
        check(arb(Q(3, 4)) * (arb(Q(7, 16)) ** Q(1, 29)) > Q(1, 2), "Left Rouche circle |y|=3/4")
        check(arb(Q(1, 5)) * (arb(Q(24, 25)) ** Q(1, 29)) < Q(1, 4)
              < arb(Q(1, 3)) * (arb(Q(8, 9)) ** Q(1, 29)), "psi(1/4) in (1/5, 1/3)")
        v0_cut = 1 - Q(1, 2 ** 16)
        check((Q(1, 5) ** 29) * Q(24, 25) > Q(1, 2 ** 75)
              and (v0_cut ** 29) * (1 - v0_cut * v0_cut) > Q(1, 2 ** 75), "Middle interval -V > 2^-75")
        check(Q(1, 2 ** 17) < (v0_cut ** 29) * (1 - v0_cut * v0_cut) < Q(1, 2 ** 14), "Right cutoff range")
        check((r * (r - 1) + (r + 1) * (r + 2)) * ((1 + Q(1, 2 ** 12)) ** (r + 2)) < 2 ** 12,
              "|V''| < 2^12 on |y-1| <= 2^-12")
        a_min = (arb(r) / (r + 2)).sqrt()
        check(-(a_min ** r) * 2 / (r + 2) < -arb(2) ** -6 and a_min > Q(9, 10), "Well depth at a")
        h_exp = -1_000_000
        eta_exp, rad_exp = -220 + 2 * h_exp, -600 + 3 * h_exp
        check(3 + 250 + rad_exp - h_exp < eta_exp - 2, "Complex angular flow stays in energy disk")
        check(12 + max(1 - 20_000_000 - 2 * rad_exp, 260 - 10_000_000 - h_exp) + 2 - eta_exp < -3_000_000,
              "Nonlinear return and derivative error < 2^-3,000,000")

        # Supplementary 1000-bit whole-oval quadrature with rigorous right-turn root enclosure
        pot = (Z ** r) * (Z * Z - 1)
        pot_deriv = pot.derivative()

        def eval_poly_arb(pol: Poly, x_val: arb) -> arb:
            acc = arb(0)
            for c_val in reversed([pol[k] for k in range(pol.degree() + 1)]):
                acc = acc * x_val + arb(c_val)
            return acc

        for left_q in (Q(-1, 2 ** 16), Q(1, 2 ** 16)):
            h_q = pot(left_q)
            lim = abs(h_q)
            # 1. Seed enclosure on I_0 = [-|h|, |h|] by exact rational endpoint sign-straddling
            #    and strict monotonicity V'(1 + I_0) > 1.
            check(pot(1 - lim) < h_q < pot(1 + lim), "Exact rational endpoint sign-straddling on seed")
            z_r = arb(-lim).union(arb(lim))
            check(eval_poly_arb(pot_deriv, 1 + z_r) > 1, "Strict monotonicity V' > 1 on seed interval")
            # 2. Strict self-map inclusion T(I_0) subset int(I_0), followed by inclusion-preserving
            #    fixed-point refinement I_{m+1} = T(I_m) enclosing z_* = T(z_*).
            z_1 = arb(h_q) / (((1 + z_r) ** r) * (2 + z_r))
            check(z_1.lower() > z_r.lower() and z_1.upper() < z_r.upper(),
                  "Strict self-map inclusion T(I_0) subset int(I_0)")
            z_r = z_1
            for _ in range(4):
                z_r = arb(h_q) / (((1 + z_r) ** r) * (2 + z_r))
            pad = arb(-1).union(arb(1)) * (arb(2) ** -992)
            right_b = 1 + z_r + pad
            # 3. Independent endpoint sign-straddling + hull monotonicity on final narrow enclosure.
            with ctx.workprec(1200):
                u_lo = arb(right_b.lower())
                u_hi = arb(right_b.upper())
                check(u_lo < u_hi and eval_poly_arb(pot_deriv, u_lo.union(u_hi)) > 1,
                      "Strict monotonicity V' > 1 on final right-turn hull")
                check(eval_poly_arb(pot, u_lo) < arb(h_q) < eval_poly_arb(pot, u_hi),
                      "Strict endpoint sign-straddling V(u_lo) < h < V(u_hi) on final enclosure")
            right_b = arb(right_b.lower()).union(arb(right_b.upper()))

            quot, rem = divmod(pot - h_q, Z - left_q)
            check(rem == Poly([]), "Exact division by left turning factor")
            fac = [arb(0)] * quot.degree()
            fac[-1] = arb(quot.leading_coefficient())
            for j_f in range(len(fac) - 1, 0, -1):
                fac[j_f - 1] = arb(quot[j_f]) + right_b * fac[j_f]
            check((right_b * fac[0] + arb(quot[0])).contains(0), "Deflated residual consistency")
            fac_poly = acb_poly(fac)
            width = right_b - arb(left_q)
            w_scale = arb(abs(h_q)) ** (arb(1) / (2 * r))
            tol = arb(2) ** (-ctx.prec + 100)
            for ep, Dp, cp, cm in dirs:
                d_acb = acb_poly([arb(Dp[k]) for k in range(Dp.degree() + 1)])

                def ig(ang: acb, analytic: bool) -> acb:
                    sn, cs = ang.sin_cos()
                    yv = acb(left_q) + acb(width) * (sn ** 2)
                    return 2 * (acb(width) ** 2) * ((sn * cs) ** 2) * fac_poly(yv).sqrt(analytic=analytic) * d_acb(yv)

                val = acb.integral(ig, 0, arb.pi() / 2, rel_tol=tol, abs_tol=tol)
                check(val.is_finite() and val.imag.contains(0), "Certified whole-oval integral")
                pred = (cp if h_q > 0 else cm) * (w_scale ** ep)
                check(abs(val.real / pred - 1) < arb(2) ** -12, f"Whole-oval quadrature at power {ep}")
                error_upper = int((abs(val.real / pred - 1).upper() * 2**36).ceil().unique_fmpz())
                check(error_upper < 2**24, "Exported relative error is strictly below 2^-12")
                quadrature.append(f"{left_q}\t{ep}\t{error_upper}")
    out = Path(__file__).resolve().parent / "outputs"
    (out / "783-brackets.tsv").write_text("\n".join(brackets) + "\n")
    (out / "783-quadrature.tsv").write_text("\n".join(quadrature) + "\n")
    print("S4: 27 exact dyadic w-brackets; normalized signs +1/8, -1/8 and slope < -Delta/2.")
    print("S4: 30 auxiliary relative integral errors < 2^-12; scalar nonlinear error bound 2^-3000000.")
    print("[2/3] Stationary count 64,005 and fully specified 783-cycle witness: OK")


# ------------------------------------------------------------------------------
# 3. Single-row interleaved binary saddle tree at M = 128 (Section B1)
# ------------------------------------------------------------------------------
def C_entry_arb(q: int, alpha: int, j: int, ell: int) -> arb:
    g0 = Q(2 * j + 1, 2 * alpha)
    binom_val = arb(prod(((g0 - m) / (m + 1) for m in range(ell)), start=Q(1)))
    exp_num = 2 * j + 1 - 2 * alpha * ell
    pow1 = exp_num + alpha - 1
    pow2 = exp_num + 2 * alpha - 1
    check(pow1 >= 0 and pow2 >= 0, "Nonnegative desingularized powers")
    rt2 = arb(2).sqrt()
    delta = rt2 - 1
    tol = arb(2) ** (-ctx.prec + 30)

    def P_q(x: acb) -> acb:
        u, acc, cur = (x * x - 1) ** 2, acb(0), acb(1)
        for _ in range(q):
            acc += cur
            cur *= u
        return acc

    def int1(t: acb, analytic: bool) -> acb:
        x = t ** alpha
        base = (2 - x * x) * P_q(x)
        if analytic and not (base.real > 0):
            return acb(arb.nan())
        return alpha * (t ** pow1) * (base ** acb(Q(exp_num, 2 * alpha)))

    def int2(t: acb, analytic: bool) -> acb:
        x = rt2 - delta * (t ** (2 * alpha))
        base = delta * (x * x) * (rt2 + x) * P_q(x)
        if analytic and not (base.real > 0):
            return acb(arb.nan())
        return 2 * alpha * delta * (t ** pow2) * (base ** acb(Q(exp_num, 2 * alpha)))

    res = acb.integral(int1, 0, 1, rel_tol=tol, abs_tol=tol) + acb.integral(int2, 0, 1, rel_tol=tol, abs_tol=tol)
    check(res.is_finite() and res.imag.contains(0), "Arb formation integral convergence")
    return binom_val * res.real


def verify_single_row_interleaved() -> None:
    with ctx.workprec(320):
        # Cross-check q = 1 against exact closed-form Beta values
        for j, ell in ((65, 0), (65, 1), (66, 0), (66, 1)):
            g0 = Q(2 * j + 1, 256)
            g = g0 - ell
            bv = arb(prod(((g0 - m) / (m + 1) for m in range(ell)), start=Q(1)))
            a, b = arb(g + Q(1, 2)), arb(g + 1)
            exact_q1 = bv * (arb(4) ** arb(g)) * (a.gamma() * b.gamma() / (a + b).gamma()) / arb(2).sqrt()
            check(C_entry_arb(1, 128, j, ell).overlaps(exact_q1), "q=1 exact Beta cross-check")

        specs = [
            (1, 65, "-5.59", "-5.58"),
            (2, 67, "-1.06", "-1.05"),
            (4, 69, "-0.354", "-0.353"),
            (8, 71, "-0.142", "-0.141"),
            (16, 73, "-0.060", "-0.059"),
        ]
        for q, j0, lo, hi in specs:
            check(64 < j0 < j0 + 1 <= 127, "Formation block avoids the protected prefix")
            check(all(1 < Q(2*j+1, 256) + Q(1, 2) < 2 for j in (j0, j0+1)),
                  "Selected columns are in the required exponent band")
            mat = arb_mat([[C_entry_arb(q, 128, j, ell) for j in (j0, j0 + 1)] for ell in (0, 1)])
            det = mat.det()
            check(not det.contains(0) and arb(lo) < det < arb(hi), f"2x2 minor (q={q}, j0={j0})")
            print(f"B1: q={q}, columns {j0},{j0+1}, determinant in ({lo},{hi})")

    # Verify h = 5 maximizes the unreserved single-row count at M = 128
    M = 128
    best_unres = max((M // (2 ** h)) * unres_opt(h, M, l) for h in range(1, 8) for l in range(1, M + 1))
    check(best_unres == 4 * unres_opt(5, 128, 128) == 58496, "Unreserved 1-row optimum at M=128 is 58,496")
    check(4 * F_native(5, 128, 128) + 9 * 128 - 40 == 59240, "Interleaved 1-row count at M=128 is 59,240")

    # Verify exact (4, 257), D = 255 rational polynomial expansion & coprimality
    X = Poly([0, 1])
    f = X
    for i in range(1, 6):
        f = f((X * X - 1) * (2 ** i))
    t4 = 8 * (X ** 4) - 8 * (X ** 2) + 1
    phi = f(t4 * 2)
    u_prime = -(phi * phi).derivative()
    check([i for i, v in enumerate(u_prime) if v] == list(range(1, 256, 2)), "128 odd x-powers in -U'(x)")
    p_y = (X ** 256 + Q(1, 5) * (X ** 4) + Q(1, 7) * (X ** 3) + Q(1, 11) * (X ** 2)).derivative()
    q_y = (X ** 2) + sum(Q(1, j + 3) * (X ** (2 * j + 1)) for j in range(128))
    p_supp = {(0, j) for j, v in enumerate(p_y) if v}
    q_supp = {(i, 0) for i, v in enumerate(u_prime) if v} | {(0, j) for j, v in enumerate(q_y) if v}
    check((len(p_supp), len(q_supp), max(i + j for i, j in p_supp | q_supp)) == (4, 257, 255),
          "Exact (4,257), D=255 single-row support and degree")
    print("[3/3] Five Arb 2x2 formation minors, single-row support sample and existential count 59,240: OK")


if __name__ == "__main__":
    verify_recurrent_and_joint()
    verify_stationary_and_witness()
    verify_single_row_interleaved()
    print("\nAll listed finite arithmetic and interval checks passed.")
