"""Stage 0: verify that this installation reproduces the published results.

Run this first. It needs no GPU, no network, and no downloaded embeddings: it
works from the canonical result files shipped in ``data/`` plus a short
self-contained quantum computation.

What it checks
--------------
1. The quantum kernel. Every feature map's fast (state-overlap) route equals the
   per-pair circuit route to machine precision. If this fails, nothing else in
   the repository can be trusted.
2. The analysis. Every headline statistic in the paper is recomputed from the
   canonical records and compared with the published value.
3. The screening rule. The flagged set, its worst gap, and the threshold sweep.

Tolerances
----------
Exact agreement is expected for the analysis, because it is recomputed from the
same stored numbers. Re-deriving features from raw embeddings on a different
scikit-learn version is a different matter and agrees to about 1e-5; see the
note in ``qks/features.py``.

Usage
-----
    python scripts/00_verify_install.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qks.analysis import (apply_fitted_model, correlate, fit_model, leave_one_domain_out,
                          leave_one_encoder_out, loo_r2, permutation_test, screen,
                          threshold_sweep)
from qks.io import load_split
from qks.kernels import self_check

#: Published values, from the paper. ``(label, expected, tolerance)``.
EXPECTED = {
    "main grid, r(A_C, gap)":        (0.873, 5e-4),
    "main grid, rho(A_C, gap)":      (0.775, 5e-4),
    "main grid, r(A_Q, gap)":        (0.898, 5e-4),
    "main grid, LOO R2 (A_C)":       (0.709, 5e-4),
    "main grid, LOO R2 (A_Q)":       (0.754, 5e-4),
    "main grid, LOO R2 (combined)":  (0.697, 5e-4),
    "screen, flagged count":         (8, 0),
    "screen, mean flagged gap":      (-0.044, 5e-4),
    "screen, worst flagged gap":     (-0.011, 5e-4),
    "LODO pooled r":                 (0.828, 5e-4),
    "LODO pooled MAE":               (0.015, 5e-4),
    "hold-out, flagged count":       (5, 0),
    "hold-out, worst flagged gap":   (0.003, 5e-4),
    "hold-out, transfer r":          (0.370, 5e-3),
}

_PASS, _FAIL = "  OK  ", " FAIL "


def _check(results: list, label: str, value: float) -> None:
    expected, tol = EXPECTED[label]
    ok = abs(value - expected) <= tol
    results.append(ok)
    print(f"[{_PASS if ok else _FAIL}] {label:<32} {value:>9.3f}   published {expected:>8.3f}")


def main() -> int:
    print("=" * 72)
    print("1. Quantum kernel implementation")
    print("=" * 72)
    self_check()

    print()
    print("=" * 72)
    print("2. Published statistics, recomputed from the canonical records")
    print("=" * 72)
    split = load_split()
    main_grid, heldout = split["main"], split["heldout"]
    print(f"loaded {len(main_grid)} main-grid and {len(heldout)} held-out configurations\n")

    A_C = [r["c_kta"] for r in main_grid]
    A_Q = [r["kta"] for r in main_grid]
    gap = [r["gap"] for r in main_grid]
    results: list[bool] = []

    c = correlate(A_C, gap)
    _check(results, "main grid, r(A_C, gap)", c["pearson_r"])
    _check(results, "main grid, rho(A_C, gap)", c["spearman_rho"])
    _check(results, "main grid, r(A_Q, gap)", correlate(A_Q, gap)["pearson_r"])
    _check(results, "main grid, LOO R2 (A_C)", loo_r2(A_C, gap))
    _check(results, "main grid, LOO R2 (A_Q)", loo_r2(A_Q, gap))

    combined = np.column_stack([A_Q,
                                [r["concentration"] for r in main_grid],
                                [r["g_exact"] for r in main_grid]])
    _check(results, "main grid, LOO R2 (combined)", loo_r2(combined, gap))

    s = screen(A_C, gap, tags=[r["tag"] for r in main_grid])
    _check(results, "screen, flagged count", s["n_flagged"])
    _check(results, "screen, mean flagged gap", s["flagged_mean_gap"])
    _check(results, "screen, worst flagged gap", s["worst_flagged_gap"])

    lodo = leave_one_domain_out(A_C, gap, [r["domain"] for r in main_grid])
    _check(results, "LODO pooled r", lodo["pooled"]["pearson_r"])
    _check(results, "LODO pooled MAE", lodo["pooled"]["mae"])

    A_C_held = [r["c_kta"] for r in heldout]
    gap_held = [r["gap"] for r in heldout]
    sh = screen(A_C_held, gap_held)
    _check(results, "hold-out, flagged count", sh["n_flagged"])
    _check(results, "hold-out, worst flagged gap", sh["worst_flagged_gap"])
    model = fit_model(A_C, gap)
    _check(results, "hold-out, transfer r", apply_fitted_model(model, A_C_held, gap_held)["pearson_r"])

    print()
    print("=" * 72)
    print("3. Supporting results")
    print("=" * 72)
    perm = permutation_test(A_C, gap)
    print(f"permutation test: {perm['n_exceeding']} of {perm['n_permutations']} "
          f"shuffles reach the observed correlation (published: none)")
    results.append(perm["n_exceeding"] == 0)

    enc = leave_one_encoder_out(A_C, gap, [r["encoder"] for r in main_grid])
    lo = min(v["pearson_r"] for v in enc.values())
    hi = max(v["pearson_r"] for v in enc.values())
    print(f"leave-one-encoder-out r: {lo:.3f} to {hi:.3f} (published 0.845 to 0.931)")
    results.append(abs(lo - 0.845) < 5e-4 and abs(hi - 0.931) < 5e-4)

    print("\nthreshold sweep on the main grid (published: worst stays at -0.011 throughout)")
    worst_ok = True
    for row in threshold_sweep(A_C, gap):
        worst_ok &= abs(row["worst_flagged_gap"] - (-0.011)) < 5e-4
        print(f"   A_C < {row['threshold']:.2f}   flagged {row['n_flagged']:>2}   "
              f"worst {row['worst_flagged_gap']:+.3f}   {row['hours_saved']:>5.1f} h saved")
    results.append(worst_ok)

    print()
    print("=" * 72)
    passed = sum(results)
    if passed == len(results):
        print(f"ALL {len(results)} CHECKS PASSED. This installation reproduces the paper.")
        return 0
    print(f"{passed} of {len(results)} checks passed. See the FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
