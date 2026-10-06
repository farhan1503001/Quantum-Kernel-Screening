"""Stage 5: the full analysis report.

Recomputes every statistic the paper reports, from the records in ``data/``, and
prints them grouped by the claim each one supports. Nothing here fits anything
to the held-out block; the model is fitted on the main grid and applied to the
hold-out unchanged.

Usage
-----
    python scripts/05_analysis.py                  # from the shipped canonical files
    python scripts/05_analysis.py --from-records   # from records you generated

``--from-records`` reads ``data/<tag>_seed42_record.json`` instead, which is how
you confirm your own rerun against the published values.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qks import io
from qks.analysis import (apply_fitted_model, correlate, fit_model, leave_one_domain_out,
                          leave_one_encoder_out, loo_r2, permutation_test, screen,
                          threshold_sweep)
from qks.config import HELDOUT_TAGS, MAIN_TAGS, SCREEN_THRESHOLD


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-records", action="store_true",
                        help="use records you generated rather than the shipped canonical files")
    args = parser.parse_args()

    if args.from_records:
        main_grid = io.load_all_records(MAIN_TAGS)
        heldout = io.load_all_records(HELDOUT_TAGS)
        print(f"loaded {len(main_grid)}/18 main-grid and {len(heldout)}/12 held-out records")
        if len(main_grid) < len(MAIN_TAGS):
            print("WARNING: the main grid is incomplete, so statistics below are partial")
    else:
        split = io.load_split()
        main_grid, heldout = split["main"], split["heldout"]
        print(f"loaded the shipped canonical files: {len(main_grid)} main, {len(heldout)} held-out")

    A_C = [r["c_kta"] for r in main_grid]
    A_Q = [r["kta"] for r in main_grid]
    gap = [r["gap"] for r in main_grid]
    tags = [r["tag"] for r in main_grid]

    # ------------------------------------------------------------------
    section("1. The landscape")
    g = np.array(gap)
    print(f"utility gap ranges from {g.min():+.4f} to {g.max():+.4f}")
    print(f"configurations where the quantum kernel wins by more than 0.005: "
          f"{int((g > 0.005).sum())}")
    print(f"exact ties (gap == 0): {int((g == 0).sum())}")
    print(f"significant deficits (interval excludes zero): "
          f"{sum(r.get('excludes_zero', False) for r in main_grid)}")

    # ------------------------------------------------------------------
    section("2. Prediction: is alignment associated with the gap?")
    for label, values in (("classical alignment A_C", A_C), ("quantum alignment A_Q", A_Q)):
        c = correlate(values, gap)
        print(f"{label}")
        print(f"   Pearson  r   = {c['pearson_r']:.3f}  (p = {c['pearson_p']:.2e})")
        print(f"   Spearman rho = {c['spearman_rho']:.3f}  (p = {c['spearman_p']:.2e})")
        print(f"   LOO R^2      = {loo_r2(values, gap):.3f}")
    perm = permutation_test(A_C, gap)
    print(f"\npermutation test on A_C: {perm['n_exceeding']} of {perm['n_permutations']} "
          f"shuffles reach the observed correlation (p {'<' if perm['n_exceeding'] == 0 else '='} "
          f"{perm['p_value']:.1e})")

    combined = np.column_stack([A_Q,
                                [r["concentration"] for r in main_grid],
                                [r["g_exact"] for r in main_grid]])
    print(f"\nadding the theory diagnostics to alignment:")
    print(f"   alignment alone LOO R^2 = {loo_r2(A_Q, gap):.3f}")
    print(f"   all three       LOO R^2 = {loo_r2(combined, gap):.3f}")

    print("\nthe theory diagnostics on their own:")
    for key, label in (("concentration", "spectral concentration"), ("g_exact", "geometric difference")):
        c = correlate([r[key] for r in main_grid], gap)
        print(f"   {label:<22} r = {c['pearson_r']:+.3f}  (p = {c['pearson_p']:.3f})")

    # ------------------------------------------------------------------
    section("3. Transfer: does it hold on data the model has not seen?")
    lodo = leave_one_domain_out(A_C, gap, [r["domain"] for r in main_grid])
    for domain, stats_ in lodo.items():
        print(f"   {domain:<12} n={stats_['n']:>2}  r={stats_['pearson_r']:+.3f}  MAE={stats_['mae']:.4f}")

    print("\nleave-one-encoder-out correlation")
    enc = leave_one_encoder_out(A_C, gap, [r["encoder"] for r in main_grid])
    for encoder, stats_ in sorted(enc.items()):
        print(f"   without {encoder:<14} n={stats_['n']:>2}  r={stats_['pearson_r']:.3f}")

    if heldout:
        A_C_held = [r["c_kta"] for r in heldout]
        gap_held = [r["gap"] for r in heldout]
        transfer = apply_fitted_model(fit_model(A_C, gap), A_C_held, gap_held)
        print(f"\nheld-out multilingual block, model fitted on the main grid and not refitted")
        print(f"   r = {transfer['pearson_r']:.3f} (p = {transfer['pearson_p']:.2f}), "
              f"MAE = {transfer['mae']:.4f}")
        print(f"   every alignment in the block is below {max(A_C_held):.3f}, a narrow band at "
              f"the low end of the fitted range")

    # ------------------------------------------------------------------
    section("4. Decision: is it safe to skip what the rule flags?")
    s = screen(A_C, gap, tags=tags)
    print(f"main grid, threshold A_C < {SCREEN_THRESHOLD}")
    print(f"   flagged {s['n_flagged']} of {s['n_total']}")
    print(f"   mean gap among flagged   {s['flagged_mean_gap']:+.4f}")
    print(f"   worst gap among flagged  {s['worst_flagged_gap']:+.4f}   <- the decisive number")
    print(f"   every flagged configuration is a deficit: {s['all_flagged_are_deficits']}")
    print(f"   mean gap among kept      {s['kept_mean_gap']:+.4f}")
    print(f"   flagged: {', '.join(s['flagged_tags'])}")

    if heldout:
        sh = screen(A_C_held, gap_held, tags=[r["tag"] for r in heldout])
        print(f"\nheld-out block, same threshold")
        print(f"   flagged {sh['n_flagged']} of {sh['n_total']}, "
              f"worst gap {sh['worst_flagged_gap']:+.4f}")
        print(f"   flagged: {', '.join(sh['flagged_tags'])}")

    print("\nthreshold sweep (main grid)")
    for row in threshold_sweep(A_C, gap):
        print(f"   A_C < {row['threshold']:.2f}   flagged {row['n_flagged']:>2}   "
              f"worst {row['worst_flagged_gap']:+.4f}   {row['hours_saved']:>5.1f} h saved")

    print()


if __name__ == "__main__":
    main()
