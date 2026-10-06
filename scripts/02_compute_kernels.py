"""Stage 2: quantum kernels, classical baselines, and the utility gap.

Reads the ``*_raw.npz`` files from stage 1, applies the fixed preprocessing, and
produces one record per configuration with every number the paper reports.

No GPU and no network are needed. The kernels are built from simulated states,
which is exact on a statevector simulator and takes seconds rather than the
83 minutes a pairwise evaluation costs. ``qks.kernels.self_check`` asserts the
equality before anything expensive runs.

Usage
-----
    python scripts/02_compute_kernels.py                  # all configurations
    python scripts/02_compute_kernels.py --tags sst2_bert
    python scripts/02_compute_kernels.py --seed 7 --no-curves

Records and kernel caches that already exist are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qks import io
from qks.config import ALL_TAGS, GRID, SEED, metric_for
from qks.diagnostics import all_diagnostics
from qks.evaluate import evaluate_configuration, learning_curve
from qks.features import check_range, prepare
from qks.kernels import quantum_kernel_matrices, self_check


def run_one(tag: str, seed: int, with_curves: bool = True, force: bool = False) -> dict:
    """Evaluate one configuration and return its record."""
    E_train, y_train, E_test, y_test = io.load_raw(tag, seed)
    X_train, y_train, X_test, y_test = prepare(E_train, y_train, E_test, y_test, seed=seed)
    check_range(X_train)
    check_range(X_test)

    if io.has_kernels(tag, seed) and not force:
        K_train, K_test, X_train, y_train, X_test, y_test = io.load_kernels(tag, seed)
    else:
        K_train, K_test = quantum_kernel_matrices(X_train, X_test)
        io.save_kernels(tag, seed, K_train, K_test, X_train, y_train, X_test, y_test)

    metric = metric_for(tag)
    record = {"tag": tag, "seed": seed, **GRID[tag]}
    record.update(evaluate_configuration(X_train, y_train, X_test, y_test,
                                         K_train, K_test, metric, seed))
    record.update(all_diagnostics(K_train, X_train, y_train))
    if with_curves:
        record["learning_curve"] = learning_curve(X_train, y_train, X_test, y_test,
                                                  K_train, K_test, metric, seed=seed)
    io.save_record(tag, seed, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="*", default=ALL_TAGS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-curves", action="store_true",
                        help="skip learning curves, which are the slow part")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("verifying the quantum kernel implementation")
    self_check()
    print()

    for tag in args.tags:
        if not io.has_raw(tag, args.seed):
            print(f"{tag:<26} SKIPPED (no embeddings; run stage 1 first)")
            continue
        if io.has_record(tag, args.seed) and not args.force:
            print(f"{tag:<26} record exists")
            continue
        t0 = time.time()
        r = run_one(tag, args.seed, with_curves=not args.no_curves, force=args.force)
        print(f"{tag:<26} gap {r['gap']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
              f"F1_Q {r['quantum_f1']:.4f} vs {r['best_classical_kernel']} "
              f"{r['best_classical_f1']:.4f}  A_C {r['c_kta']:.4f}  ({time.time() - t0:.0f} s)")

    done = sum(io.has_record(t, args.seed) for t in args.tags)
    print(f"\n{done} of {len(args.tags)} records present for seed {args.seed}")


if __name__ == "__main__":
    main()
