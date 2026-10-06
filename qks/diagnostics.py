"""The three candidate predictors of the utility gap.

* ``kta`` -- centered kernel-target alignment (Cristianini et al. 2001; Cortes
  et al. 2012). This is the quantity the paper's screening rule is built on.
  Computed on a classical RBF Gram matrix it needs no quantum computation.
* ``concentration`` -- normalized effective rank, the spectral signature of
  exponential concentration (Thanasilp et al. 2024).
* ``geometric_difference`` -- the test of Huang et al. (2021) for whether a
  quantum kernel reaches geometry a classical kernel cannot reproduce.

All three take a Gram matrix and, where labels are needed, the training labels.
Nothing here needs a test split or a trained model.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel


def kta(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment between a Gram matrix and the labels.

    The ideal kernel is ``Y[i, j] = 1`` when ``y[i] == y[j]`` and 0 otherwise,
    which extends the original binary construction to the multiclass case. Both
    matrices are centered by ``H = I - (1/n) 11^T`` before the Frobenius inner
    product, as in Cortes et al. (2012).

    Returns a value in ``[0, 1]``; higher means the kernel geometry agrees more
    closely with the label structure.
    """
    y = np.asarray(y)
    n = len(y)
    Y = np.zeros((n, n))
    for c in np.unique(y):
        m = (y == c).astype(float)
        Y += np.outer(m, m)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Yc = H @ K @ H, H @ Y @ H
    return float(np.sum(Kc * Yc) / np.sqrt(np.sum(Kc * Kc) * np.sum(Yc * Yc)))


def concentration(K: np.ndarray) -> float:
    """Normalized effective rank of a Gram matrix, in ``[1/n, 1]``.

    This locates a kernel between two degenerate extremes, both of which destroy
    learnability. Entries collapsed toward a common value approach rank one and
    ``rho -> 1/n``; off-diagonal entries collapsed toward zero approach the
    identity and ``rho -> 1``. Informative class structure lives between the
    poles, so this is a descriptor of spectral shape, not a monotone score.
    """
    mu = np.maximum(np.linalg.eigvalsh(K), 0)
    lam = mu / mu.sum()
    return float(np.exp(-np.sum(lam * np.log(lam + 1e-15))) / K.shape[0])


def geometric_difference(K_quantum: np.ndarray, K_classical: np.ndarray,
                         ridge: float = 1e-6) -> float:
    """Exact geometric difference ``g(K_C || K_Q)``.

    Both matrices are trace-normalized to ``n`` first. Because ``K_C`` is badly
    ill-conditioned on concentrated kernels, the inverse is ridge-regularized as
    ``(K_C + ridge * n * I)^-1``.

    The magnitude of ``g`` depends strongly on ``ridge``; sweep it with
    ``geometric_difference_sweep`` before drawing any conclusion from its value.
    """
    n = K_quantum.shape[0]
    KQ = K_quantum * n / np.trace(K_quantum)
    KC = K_classical * n / np.trace(K_classical)
    w, V = np.linalg.eigh(KQ)
    w = np.maximum(w, 0)
    sqrt_KQ = (V * np.sqrt(w)) @ V.T
    inner = sqrt_KQ @ np.linalg.inv(KC + ridge * n * np.eye(n)) @ sqrt_KQ
    return float(np.sqrt(max(np.linalg.eigvalsh(inner).max(), 0.0)))


def geometric_difference_sweep(K_quantum, K_classical,
                               ridges=(1e-4, 1e-5, 1e-6, 1e-7)) -> dict[str, float]:
    """``g`` at several ridge strengths, keyed by the ridge in ``%.0e`` form."""
    return {f"{r:.0e}": geometric_difference(K_quantum, K_classical, r) for r in ridges}


def effective_dimension_ratio(K_quantum: np.ndarray, K_classical: np.ndarray) -> float:
    """A cheap effective-dimension proxy for ``g``, reported in the appendix only."""
    eff = lambda K: (np.trace(K) ** 2) / (np.sum(K ** 2) + 1e-10)
    return float(eff(K_quantum) / (eff(K_classical) + 1e-10))


def mean_offdiagonal(K: np.ndarray) -> float:
    """Mean of the off-diagonal entries, the direct measure of concentration."""
    off = ~np.eye(K.shape[0], dtype=bool)
    return float(K[off].mean())


# --------------------------------------------------------------------------
# Classical Gram matrices used for alignment
# --------------------------------------------------------------------------


def classical_gram(X: np.ndarray, kind: str = "rbf_1/d") -> np.ndarray:
    """Gram matrix for one of the six classical kernels used in the robustness study.

    ``rbf_1/d`` is the paper's choice and the one the screening threshold refers
    to. ``rbf_scale`` matches scikit-learn's ``gamma='scale'`` heuristic.
    """
    d = X.shape[1]
    if kind == "linear":
        return linear_kernel(X, X)
    if kind == "poly3":
        return polynomial_kernel(X, X, degree=3, coef0=1, gamma=1.0 / d)
    if kind == "rbf_0.5/d":
        return rbf_kernel(X, X, gamma=0.5 / d)
    if kind == "rbf_1/d":
        return rbf_kernel(X, X, gamma=1.0 / d)
    if kind == "rbf_2/d":
        return rbf_kernel(X, X, gamma=2.0 / d)
    if kind == "rbf_scale":
        return rbf_kernel(X, X, gamma=1.0 / (d * X.var()))
    raise ValueError(f"unknown classical kernel: {kind!r}")


#: The six classical kernels, in the order they appear in the robustness table.
CLASSICAL_KERNELS = ("rbf_1/d", "rbf_0.5/d", "rbf_2/d", "rbf_scale", "linear", "poly3")


def alignment_variants(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Classical alignment under every kernel in ``CLASSICAL_KERNELS``."""
    return {k: kta(classical_gram(X, k), y) for k in CLASSICAL_KERNELS}


def classical_alignment(X: np.ndarray, y: np.ndarray) -> float:
    """The screening quantity ``A_C``: alignment of the RBF kernel at gamma = 1/d.

    This is the whole cost of the screening rule. It needs the training features
    and labels and no quantum computation at all.
    """
    return kta(classical_gram(X, "rbf_1/d"), y)


def all_diagnostics(K_quantum: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    """Every diagnostic for one configuration, as stored in the canonical files."""
    K_rbf = classical_gram(X, "rbf_1/d")
    return {
        "kta": kta(K_quantum, y),
        "c_kta": kta(K_rbf, y),
        "concentration": concentration(K_quantum),
        "g_exact": geometric_difference(K_quantum, K_rbf),
        "g_proxy": effective_dimension_ratio(K_quantum, K_rbf),
        "g_lambda_sweep": geometric_difference_sweep(K_quantum, K_rbf),
        "offdiag_quantum": mean_offdiagonal(K_quantum),
        "offdiag_rbf": mean_offdiagonal(K_rbf),
    }
