"""Classical baselines, the utility gap, and its bootstrap interval.

The utility gap of a configuration is

    Delta = F1_quantum - max(F1_linear, F1_rbf, F1_poly)

on the same eight scaled features, with every SVM at the shared regularization
``C = 1``. Taking the maximum over three baselines makes the comparison
deliberately conservative: the quantum kernel has to beat the strongest
classical competitor for ``Delta`` to be positive, not merely an average one.

``C = 1`` is not tuned in favour of either side. It is the scikit-learn default
and is applied identically to the quantum and classical models.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from .config import C_SVM, N_BOOTSTRAP, SEED

#: The three classical baselines, in a fixed order.
CLASSICAL_BASELINES = ("Linear", "RBF", "Polynomial")


def f1_scorer(metric: str):
    """Return an F1 function for 'binary' or 'macro' averaging."""
    average = "binary" if metric == "binary" else "macro"
    return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average=average))


def _make_classifier(name: str, seed: int):
    if name == "Linear":
        return SVC(kernel="linear", C=C_SVM, random_state=seed)
    if name == "RBF":
        return SVC(kernel="rbf", C=C_SVM, gamma="scale", random_state=seed)
    if name == "Polynomial":
        return SVC(kernel="poly", degree=3, coef0=1, C=C_SVM, gamma="scale", random_state=seed)
    raise ValueError(f"unknown baseline: {name!r}")


def classical_baselines(X_train, y_train, X_test, y_test, metric: str, seed: int = SEED):
    """Fit the three classical SVMs. Returns ``(scores, fitted_models)``.

    ``scores`` maps each baseline name to its accuracy and F1.
    """
    f1 = f1_scorer(metric)
    scores, fitted = {}, {}
    for name in CLASSICAL_BASELINES:
        clf = _make_classifier(name, seed)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        fitted[name] = clf
        scores[name] = {"accuracy": float(accuracy_score(y_test, y_pred)),
                        "f1": f1(y_test, y_pred)}
    return scores, fitted


def quantum_svm(K_train, y_train, K_test):
    """Fit an SVM on a precomputed quantum kernel and predict the test split."""
    return SVC(kernel="precomputed", C=C_SVM).fit(K_train, y_train).predict(K_test)


def bootstrap_gap_interval(y_test, y_quantum, y_classical, metric: str,
                           seed: int = SEED, n_resamples: int = N_BOOTSTRAP):
    """Percentile bootstrap interval for the gap, resampling the test set.

    Both models are scored on the *same* resampled indices each time, so the
    interval is for the paired difference and not for two independent scores.
    Resamples that lose a class are skipped, because F1 is undefined there.

    Returns ``(lo, hi, excludes_zero)``.
    """
    f1 = f1_scorer(metric)
    rng = np.random.RandomState(seed)
    y_test = np.asarray(y_test)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.randint(0, len(y_test), len(y_test))
        if len(np.unique(y_test[idx])) < 2:
            continue
        deltas.append(f1(y_test[idx], y_quantum[idx]) - f1(y_test[idx], y_classical[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi), bool(lo > 0 or hi < 0)


def evaluate_configuration(X_train, y_train, X_test, y_test, K_train, K_test,
                           metric: str, seed: int = SEED) -> dict:
    """Score one configuration end to end and return the record the paper uses.

    The returned dictionary has the same field names as the released
    ``*_grid_record.json`` files, so records produced here merge directly into
    the canonical data.
    """
    f1 = f1_scorer(metric)
    scores, fitted = classical_baselines(X_train, y_train, X_test, y_test, metric, seed)
    best_name = max(scores, key=lambda k: scores[k]["f1"])
    best_f1 = scores[best_name]["f1"]

    y_quantum = quantum_svm(K_train, y_train, K_test)
    quantum_f1 = f1(y_test, y_quantum)
    gap = quantum_f1 - best_f1

    y_classical = fitted[best_name].predict(X_test)
    lo, hi, excludes_zero = bootstrap_gap_interval(
        y_test, y_quantum, y_classical, metric, seed)

    return {
        "quantum_f1": quantum_f1,
        "quantum_accuracy": float(accuracy_score(y_test, y_quantum)),
        "best_classical_f1": best_f1,
        "best_classical_kernel": best_name,
        "gap": gap,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": excludes_zero,
        "classical_baselines": scores,
        "metric": metric,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def learning_curve(X_train, y_train, X_test, y_test, K_train, K_test, metric: str,
                   sizes=(50, 100, 200, 400, 700, 1000), seed: int = SEED) -> dict:
    """F1 against training-set size for the quantum, linear, and RBF kernels.

    Each size uses the first ``n`` training points, so the curves are nested and
    the comparison at every ``n`` is on identical data. The test split is fixed.
    """
    f1 = f1_scorer(metric)
    out = {"sizes": list(sizes), "quantum": [], "linear": [], "rbf": [], "metric": metric}
    for n in sizes:
        out["quantum"].append(f1(y_test, quantum_svm(K_train[:n, :n], y_train[:n], K_test[:, :n])))
        out["linear"].append(f1(y_test, _make_classifier("Linear", seed)
                                .fit(X_train[:n], y_train[:n]).predict(X_test)))
        out["rbf"].append(f1(y_test, _make_classifier("RBF", seed)
                             .fit(X_train[:n], y_train[:n]).predict(X_test)))
    return out
