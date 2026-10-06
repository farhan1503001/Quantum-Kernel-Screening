"""Preprocessing from raw encoder embeddings to the eight scaled features.

Every configuration goes through exactly this path, which is what makes the
grid comparable:

1. PCA to ``N_QUBITS`` components, fitted on the training pool only.
2. Min-max scaling to ``[0, pi]``, fitted on the training pool only, with test
   values clipped to the same interval.
3. A fixed-seed draw of ``N_TRAIN`` and ``N_TEST`` points from the pools.

Step 2 is a substantive choice rather than bookkeeping. The bandwidth of the
data encoding governs both the expressivity and the concentration behavior of a
quantum kernel, so fixing the input range across all configurations holds that
factor constant while the data and the encoder vary.

The PCA seed matters because scikit-learn uses a randomized solver for matrices
of this shape. It is passed explicitly so the whole preprocessing path is a pure
function of ``seed``.

Reproducibility note. Re-running this preprocessing on a different scikit-learn
version reproduces the published features to about 1e-5 rather than exactly,
because the randomized solver differs between releases. That difference does not
move the F1 scores or the utility gap at all, since an SVM decision is not
sensitive at that scale, and it moves the diagnostics by about 1e-7. The one
quantity that amplifies it is the geometric difference, whose ridge-regularized
inverse turns 1e-5 in the features into about 3e-3 in ``g``. Expected tolerances
are asserted in ``scripts/00_verify_install.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .config import FEATURE_RANGE, N_QUBITS, N_TEST, N_TRAIN, SEED


def to_features(E_train: np.ndarray, E_test: np.ndarray, seed: int = SEED,
                n_components: int = N_QUBITS):
    """PCA then min-max scaling. Returns the full pools, not yet subsampled.

    Both transforms are fitted on the training pool alone. Test values are
    clipped to ``[0, pi]`` because a test point can fall outside the training
    range after PCA, and a feature outside that interval would change the
    meaning of the encoding angle.
    """
    # float64 throughout: encoders emit float32, and sklearn's randomized PCA
    # solver preserves the input dtype. Casting up makes the result deterministic
    # across BLAS implementations at a negligible memory cost.
    E_train = np.asarray(E_train, dtype=np.float64)
    E_test = np.asarray(E_test, dtype=np.float64)
    pca = PCA(n_components=n_components, random_state=seed).fit(E_train)
    P_train, P_test = pca.transform(E_train), pca.transform(E_test)
    scaler = MinMaxScaler(feature_range=FEATURE_RANGE).fit(P_train)
    lo, hi = FEATURE_RANGE
    X_train = np.clip(scaler.transform(P_train), lo, hi)
    X_test = np.clip(scaler.transform(P_test), lo, hi)
    return X_train, X_test


def subsample(X_train, y_train, X_test, y_test, seed: int = SEED,
              n_train: int = N_TRAIN, n_test: int = N_TEST):
    """Draw the final ``n_train`` and ``n_test`` points from the pools.

    Two separate ``RandomState(seed)`` generators are used, one per split, which
    is what the original pipeline did. Pools already at the target size are
    returned unchanged rather than resampled.
    """
    if len(X_train) > n_train:
        idx = np.random.RandomState(seed).choice(len(X_train), n_train, replace=False)
        X_train, y_train = X_train[idx], np.asarray(y_train)[idx]
    if len(X_test) > n_test:
        idx = np.random.RandomState(seed).choice(len(X_test), n_test, replace=False)
        X_test, y_test = X_test[idx], np.asarray(y_test)[idx]
    return X_train, y_train, X_test, y_test


def prepare(E_train, y_train, E_test, y_test, seed: int = SEED):
    """Full path from raw embeddings to the scaled, subsampled features.

    This is the single entry point used by the kernel and seed-study scripts, so
    that no caller can accidentally reorder the steps.
    """
    X_train, X_test = to_features(E_train, E_test, seed=seed)
    return subsample(X_train, y_train, X_test, y_test, seed=seed)


def check_range(X: np.ndarray, tol: float = 1e-5) -> None:
    """Assert that features lie in ``[0, pi]``. Called before any circuit runs.

    The default tolerance accommodates features that were computed in float32
    and stored as float64, which overshoots ``pi`` by about 3e-7. It is still
    tight enough to catch a genuinely unscaled or misscaled feature matrix.
    """
    lo, hi = FEATURE_RANGE
    assert X.min() >= lo - tol and X.max() <= hi + tol, (
        f"features outside {FEATURE_RANGE}: got [{X.min():.4f}, {X.max():.4f}]")
