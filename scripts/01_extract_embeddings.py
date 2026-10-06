"""Stage 1: extract frozen encoder embeddings for every configuration.

This is the only stage that needs a GPU and network access, and it is by far the
slowest. Everything downstream reads the ``*_raw.npz`` files it writes, so this
runs once and the rest of the pipeline can be rerun freely.

Usage
-----
    python scripts/01_extract_embeddings.py                 # all 30 configurations
    python scripts/01_extract_embeddings.py --tags sst2_bert cifar10
    python scripts/01_extract_embeddings.py --seeds 42 7 123   # for the seed study

Configurations whose file already exists are skipped, so an interrupted run can
simply be restarted.

Runtime: roughly 30 to 60 minutes for the full grid on a T4, most of it dataset
download rather than encoding.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from qks import io
from qks.config import ALL_TAGS, GRID, SEED
from qks.datasets import LOADERS
from qks.encoders import EXTRACTORS, prepare_torch

#: Encoders that consume (image, text) pairs rather than a single modality.
JOINT_ENCODERS = {"CLIP-joint", "SigLIP-joint", "FLAVA-joint"}


def extract_one(tag: str, seed: int) -> None:
    """Load the data for one configuration, encode it, and store the embeddings."""
    spec = GRID[tag]
    dataset, encoder = spec["dataset"], spec["encoder"]
    loader, extractor = LOADERS[dataset], EXTRACTORS[encoder]

    # the multilingual loader takes a language rather than a seed-only signature
    if dataset == "sentiment":
        train_x, y_train, test_x, y_test = loader(spec["language"], seed)
    else:
        train_x, y_train, test_x, y_test = loader(seed)

    if encoder in JOINT_ENCODERS:
        # pairs arrive as [(image, text), ...]; encode the towers then concatenate
        tr_images, tr_texts = zip(*train_x)
        te_images, te_texts = zip(*test_x)
        if encoder == "SigLIP-joint":
            # siglip_joint already returns the concatenation
            E_train = extractor(list(tr_images), list(tr_texts))
            E_test = extractor(list(te_images), list(te_texts))
        else:
            tr_img, tr_txt = extractor(list(tr_images), list(tr_texts))
            te_img, te_txt = extractor(list(te_images), list(te_texts))
            E_train = np.hstack([tr_img, tr_txt])
            E_test = np.hstack([te_img, te_txt])
    else:
        E_train = extractor(train_x)
        E_test = extractor(test_x)

    io.save_raw(tag, seed, E_train, y_train, E_test, y_test)
    print(f"  saved E_tr {E_train.shape}  E_te {E_test.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="*", default=ALL_TAGS,
                        help="configuration tags to extract (default: all 30)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[SEED],
                        help="seeds to extract (default: 42, the published seed)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract even if the file already exists")
    args = parser.parse_args()

    prepare_torch()
    queued = [(t, s) for t in args.tags for s in args.seeds
              if args.force or not io.has_raw(t, s)]
    print(f"{len(queued)} of {len(args.tags) * len(args.seeds)} "
          f"(configuration, seed) pairs to extract\n")

    for tag, seed in queued:
        print(f"=== {tag} / seed {seed} ===")
        t0 = time.time()
        try:
            extract_one(tag, seed)
            print(f"  done in {(time.time() - t0) / 60:.1f} min")
        except Exception as exc:                               # pragma: no cover
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    missing = [(t, s) for t in args.tags for s in args.seeds if not io.has_raw(t, s)]
    print("\nall requested embeddings present" if not missing else f"\nstill missing: {missing}")


if __name__ == "__main__":
    main()
