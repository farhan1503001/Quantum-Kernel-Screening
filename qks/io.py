"""Reading and writing the study's artifacts.

Two file kinds are used throughout:

``<tag>_seed<seed>_raw.npz``
    Raw encoder embeddings before PCA, with ``E_tr``, ``y_tr``, ``E_te``,
    ``y_te``. Written by the extraction script; the only artifact that needs a
    GPU and network access to produce.

``<tag>_seed<seed>_record.json``
    One configuration's results: F1 scores, the gap and its interval, the
    diagnostics. Field names match the released canonical files, so records
    produced here merge directly into ``grid30_canonical.json``.

Kernel matrices are cached as ``<tag>_seed<seed>_kernels.npz`` so a rerun never
recomputes them.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import DATA_DIR, SEED


def _path(name: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


# --------------------------------------------------------------------------
# Raw embeddings
# --------------------------------------------------------------------------


def raw_path(tag: str, seed: int = SEED) -> str:
    return _path(f"{tag}_seed{seed}_raw.npz")


def save_raw(tag: str, seed: int, E_train, y_train, E_test, y_test) -> str:
    """Store raw encoder embeddings before PCA."""
    p = raw_path(tag, seed)
    np.savez_compressed(p, E_tr=E_train, y_tr=y_train, E_te=E_test, y_te=y_test)
    return p


def load_raw(tag: str, seed: int = SEED):
    """Return ``(E_train, y_train, E_test, y_test)``."""
    z = np.load(raw_path(tag, seed), allow_pickle=False)
    return z["E_tr"], z["y_tr"], z["E_te"], z["y_te"]


def has_raw(tag: str, seed: int = SEED) -> bool:
    return os.path.exists(raw_path(tag, seed))


# --------------------------------------------------------------------------
# Kernel matrices
# --------------------------------------------------------------------------


def kernel_path(tag: str, seed: int = SEED) -> str:
    return _path(f"{tag}_seed{seed}_kernels.npz")


def save_kernels(tag: str, seed: int, K_train, K_test, X_train, y_train, X_test, y_test) -> str:
    """Cache the quantum kernels together with the features that produced them."""
    p = kernel_path(tag, seed)
    np.savez_compressed(p, K_train_quantum=K_train, K_test_quantum=K_test,
                        X_tr=X_train, y_tr=y_train, X_te=X_test, y_te=y_test)
    return p


def load_kernels(tag: str, seed: int = SEED):
    """Return ``(K_train, K_test, X_train, y_train, X_test, y_test)``."""
    z = np.load(kernel_path(tag, seed), allow_pickle=False)
    return (z["K_train_quantum"], z["K_test_quantum"],
            z["X_tr"], z["y_tr"], z["X_te"], z["y_te"])


def has_kernels(tag: str, seed: int = SEED) -> bool:
    return os.path.exists(kernel_path(tag, seed))


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def record_path(tag: str, seed: int = SEED) -> str:
    return _path(f"{tag}_seed{seed}_record.json")


def save_record(tag: str, seed: int, record: dict) -> str:
    p = record_path(tag, seed)
    with open(p, "w") as fh:
        json.dump(record, fh, indent=2)
    return p


def load_record(tag: str, seed: int = SEED) -> dict:
    with open(record_path(tag, seed)) as fh:
        return json.load(fh)


def has_record(tag: str, seed: int = SEED) -> bool:
    return os.path.exists(record_path(tag, seed))


def load_all_records(tags, seed: int = SEED) -> list[dict]:
    """Load every record that exists, skipping the ones that do not."""
    return [load_record(t, seed) for t in tags if has_record(t, seed)]


# --------------------------------------------------------------------------
# Canonical files shipped with the repository
# --------------------------------------------------------------------------


def load_canonical(name: str = "grid30_canonical.json") -> list[dict]:
    """Load one of the released canonical result files from ``data/``."""
    with open(_path(name)) as fh:
        return json.load(fh)


def load_split(name: str = "grid_split.json") -> dict:
    """Load the main / held-out split, a dict with ``main`` and ``heldout`` keys."""
    with open(_path(name)) as fh:
        return json.load(fh)
