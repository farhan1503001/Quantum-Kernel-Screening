"""Stage 3: the screen under six alternative quantum kernels.

Recomputes the utility gap of every configuration under each feature map in
``qks.kernels.FEATURE_MAPS``, on the identical features. Classical alignment is
unchanged by construction, so the same configurations are flagged in every case
and what varies is whether they deserved to be.

This is the experiment behind the paper's kernel-robustness table and figure. It
needs stage 2 to have run, because it reuses the cached features.

Usage
-----
    python scripts/03_ablation_featuremaps.py
    python scripts/03_ablation_featuremaps.py --tags sst2_bert cifar10

Runtime: about 15 minutes for all 30 configurations on a CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qks import io
from qks.config import ALL_TAGS, GRID, SEED, metric_for, path
from qks.diagnostics import kta, mean_offdiagonal
from qks.evaluate import f1_scorer, quantum_svm
from qks.kernels import FEATURE_MAPS, make_fast_kernel, self_check

OUTPUT = "featuremap_ablation.json"


def ablate_one(tag: str, seed: int = SEED) -> dict:
    """Every feature map's gap for one configuration, on its cached features."""
    K_train, K_test, X_train, y_train, X_test, y_test = io.load_kernels(tag, seed)
    record = io.load_record(tag, seed)
    best_classical = record["best_classical_f1"]
    f1 = f1_scorer(metric_for(tag))

    out = {}
    for name, (fmap, n_wires) in FEATURE_MAPS.items():
        kernel = make_fast_kernel(fmap, n_wires)
        K_tr, K_te = kernel(X_train), kernel(X_test, X_train)
        quantum_f1 = f1(y_test, quantum_svm(K_tr, y_train, K_te))
        out[name] = {
            "quantum_f1": quantum_f1,
            "gap": quantum_f1 - best_classical,
            "kta": kta(K_tr, y_train),
            "offdiag": mean_offdiagonal(K_tr),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="*", default=ALL_TAGS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print("verifying every feature map before the long run")
    self_check()
    print()

    out_path = path(OUTPUT)
    results = {}
    try:
        with open(out_path) as fh:
            results = json.load(fh)
    except FileNotFoundError:
        pass

    for tag in args.tags:
        if not (io.has_kernels(tag, args.seed) and io.has_record(tag, args.seed)):
            print(f"{tag:<26} SKIPPED (run stage 2 first)")
            continue
        if tag in results and not args.force:
            print(f"{tag:<26} already done")
            continue
        t0 = time.time()
        kernels = ablate_one(tag, args.seed)
        record = io.load_record(tag, args.seed)
        results[tag] = {"tag": tag, "domain": GRID[tag]["domain"],
                        "encoder": GRID[tag]["encoder"], "c_kta": record["c_kta"],
                        "best_classical_f1": record["best_classical_f1"],
                        "kernels": kernels}
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1)
        gaps = "  ".join(f"{n} {kernels[n]['gap']:+.3f}" for n in FEATURE_MAPS)
        print(f"{tag:<26} {gaps}   ({time.time() - t0:.0f} s)")

    print(f"\n{len(results)} configurations written to {out_path}")


if __name__ == "__main__":
    main()
