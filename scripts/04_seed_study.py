"""Stage 4: seed variability at the full sample size.

Reruns a subset of configurations under additional seeds with an otherwise
identical pipeline, to separate real differences between configurations from
the variability of a single draw.

The seed changes the examples drawn from each source where the source allows it,
the PCA fit, and the classifiers. Nothing else moves. For CIFAR-10, MSCOCO, and
SNLI-VE the candidate pool is fixed by dataset order, so the seeds resample
within a fixed pool and their spread understates full resampling variance; this
is reported alongside the numbers rather than hidden.

Stage 1 must have been run with the extra seeds first:

    python scripts/01_extract_embeddings.py --tags sst2_bert ... --seeds 42 7 123

Usage
-----
    python scripts/04_seed_study.py
    python scripts/04_seed_study.py --tags sst2_bert --seeds 42 7 123
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qks import io
from qks.config import SCREEN_THRESHOLD, SEEDS_STUDY

#: The six configurations rerun in the paper, two per domain, chosen to span the
#: range of gaps rather than to favour any outcome.
DEFAULT_TAGS = ["sst2_bert", "agnews_roberta", "cifar10",
                "fashion_vit", "mscoco_flava", "snlive_siglip"]

#: Configurations whose candidate pool is fixed by dataset order.
FIXED_POOL = {"cifar10", "mscoco_flava", "snlive_siglip"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS_STUDY))
    args = parser.parse_args()

    print(f"{'configuration':<18}" + "".join(f"{'seed ' + str(s):>10}" for s in args.seeds)
          + f"{'mean':>9}{'sd':>8}   same side of threshold")
    print("-" * (18 + 10 * len(args.seeds) + 17 + 24))

    spreads = []
    for tag in args.tags:
        records = [io.load_record(tag, s) for s in args.seeds if io.has_record(tag, s)]
        if len(records) < len(args.seeds):
            missing = [s for s in args.seeds if not io.has_record(tag, s)]
            print(f"{tag:<18} incomplete: missing seeds {missing}")
            continue
        gaps = np.array([r["gap"] for r in records])
        alignments = np.array([r["c_kta"] for r in records])
        same_side = bool((alignments < SCREEN_THRESHOLD).all()
                         or (alignments >= SCREEN_THRESHOLD).all())
        spreads.append(gaps.std(ddof=1))
        note = " (fixed pool)" if tag in FIXED_POOL else ""
        print(f"{tag:<18}" + "".join(f"{g:>+10.3f}" for g in gaps)
              + f"{gaps.mean():>+9.3f}{gaps.std(ddof=1):>8.3f}   {str(same_side):<6}{note}")

    if spreads:
        print(f"\nmean gap standard deviation across configurations: {np.mean(spreads):.4f}")
        print("A configuration staying on the same side of the threshold under every seed")
        print("means the screening decision does not depend on the draw.")


if __name__ == "__main__":
    main()
