"""Statistical analysis of the grid: does alignment predict the gap?

Three questions are kept separate throughout, because they are different claims
supported by different evidence:

*prediction*  -- is alignment associated with the gap, and does a model fitted
                 on it predict configurations it has not seen?
                 ``correlate``, ``permutation_test``, ``loo_r2``.
*transfer*    -- does that hold across domains and on the held-out block?
                 ``leave_one_domain_out``, ``apply_fitted_model``.
*decision*    -- is it safe to skip a configuration the rule flags?
                 ``screen``.

The decision claim is the one the paper rests on. A screen is useful when
nothing it flags was worth running, which is ``worst_flagged_gap``, not when its
correlation is high.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

from .config import SCREEN_THRESHOLD


def correlate(x, y) -> dict:
    """Pearson and Spearman correlation with p-values."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return {"pearson_r": float(r), "pearson_p": float(p_r),
            "spearman_rho": float(rho), "spearman_p": float(p_rho), "n": len(x)}


def permutation_test(x, y, n_permutations: int = 10_000, seed: int = 42) -> dict:
    """Permutation test on the Pearson correlation.

    Shuffles ``y`` and counts how often the shuffled absolute correlation
    reaches the observed one. Reports the conservative ``(count + 1) / (n + 1)``
    p-value, so a result that is never reached is reported as ``< 1/(n+1)``
    rather than as exactly zero.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = abs(stats.pearsonr(x, y)[0])
    rng = np.random.RandomState(seed)
    null = np.array([abs(stats.pearsonr(x, rng.permutation(y))[0])
                     for _ in range(n_permutations)])
    count = int((null >= observed).sum())
    return {"observed_r": float(observed), "n_exceeding": count,
            "n_permutations": n_permutations,
            "p_value": (count + 1) / (n_permutations + 1)}


def loo_r2(X, y) -> float:
    """Leave-one-out cross-validated R^2 of a linear model.

    This measures prediction rather than association: every point is predicted
    by a model that never saw it. ``X`` may be one column (alignment alone) or
    several (alignment combined with other diagnostics).
    """
    X = np.asarray(X, float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    y = np.asarray(y, float)
    predictions = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    return float(r2_score(y, predictions))


def leave_one_domain_out(alignment, gap, domains) -> dict:
    """Fit on all but one domain, predict the held-out domain.

    Guards against the model reading domain identity rather than a relationship
    that holds within a domain. Returns per-domain and pooled statistics.
    """
    alignment = np.asarray(alignment, float)
    gap = np.asarray(gap, float)
    domains = np.asarray(domains)
    pooled = np.zeros(len(gap))
    per_domain = {}
    for domain in np.unique(domains):
        test = domains == domain
        model = LinearRegression().fit(alignment[~test].reshape(-1, 1), gap[~test])
        predicted = model.predict(alignment[test].reshape(-1, 1))
        pooled[test] = predicted
        per_domain[str(domain)] = {
            "n": int(test.sum()),
            "pearson_r": float(stats.pearsonr(predicted, gap[test])[0]),
            "mae": float(np.abs(predicted - gap[test]).mean()),
        }
    per_domain["pooled"] = {
        "n": len(gap),
        "pearson_r": float(stats.pearsonr(pooled, gap)[0]),
        "mae": float(np.abs(pooled - gap).mean()),
    }
    return per_domain


def leave_one_encoder_out(alignment, gap, encoders) -> dict:
    """Correlation after removing every configuration that uses one encoder.

    Shows whether any single encoder carries the relationship.
    """
    alignment = np.asarray(alignment, float)
    gap = np.asarray(gap, float)
    encoders = np.asarray(encoders)
    out = {}
    for encoder in np.unique(encoders):
        keep = encoders != encoder
        out[str(encoder)] = {"n": int(keep.sum()),
                             "pearson_r": float(stats.pearsonr(alignment[keep], gap[keep])[0])}
    return out


def fit_model(alignment, gap) -> LinearRegression:
    """Fit the linear alignment-to-gap model used for transfer tests."""
    return LinearRegression().fit(np.asarray(alignment, float).reshape(-1, 1),
                                  np.asarray(gap, float))


def apply_fitted_model(model: LinearRegression, alignment, gap) -> dict:
    """Apply a model fitted elsewhere to a new block, without refitting."""
    alignment = np.asarray(alignment, float)
    gap = np.asarray(gap, float)
    predicted = model.predict(alignment.reshape(-1, 1))
    r, p = stats.pearsonr(predicted, gap)
    return {"n": len(gap), "pearson_r": float(r), "pearson_p": float(p),
            "mae": float(np.abs(predicted - gap).mean())}


def screen(alignment, gap, threshold: float = SCREEN_THRESHOLD, tags=None) -> dict:
    """Apply the screening rule and report what it would have cost.

    ``worst_flagged_gap`` is the decisive number: the most a practitioner
    following the rule would have given up. A screen is correct when this is at
    or below zero, meaning every configuration it told you to skip was indeed
    not worth running.
    """
    alignment = np.asarray(alignment, float)
    gap = np.asarray(gap, float)
    flagged = alignment < threshold
    result = {
        "threshold": threshold,
        "n_total": int(len(gap)),
        "n_flagged": int(flagged.sum()),
        "flagged_mean_gap": float(gap[flagged].mean()) if flagged.any() else float("nan"),
        "worst_flagged_gap": float(gap[flagged].max()) if flagged.any() else float("nan"),
        "kept_mean_gap": float(gap[~flagged].mean()) if (~flagged).any() else float("nan"),
        "all_flagged_are_deficits": bool((gap[flagged] <= 0).all()) if flagged.any() else True,
    }
    if tags is not None:
        result["flagged_tags"] = [t for t, f in zip(tags, flagged) if f]
    return result


def threshold_sweep(alignment, gap, thresholds=(0.02, 0.03, 0.05, 0.08, 0.10, 0.15),
                    minutes_per_kernel: float = 83.0) -> list[dict]:
    """Sweep the screening threshold and report flagged count, worst gap, and time saved."""
    rows = []
    for t in thresholds:
        s = screen(alignment, gap, t)
        s["hours_saved"] = s["n_flagged"] * minutes_per_kernel / 60.0
        rows.append(s)
    return rows


def summarize_grid(records: list[dict], alignment_key: str = "c_kta") -> dict:
    """Headline statistics for a list of configuration records.

    Expects each record to have ``gap``, the alignment key, ``domain``, and
    ``encoder``, which is the schema of the released canonical files.
    """
    gap = np.array([r["gap"] for r in records], float)
    alignment = np.array([r[alignment_key] for r in records], float)
    tags = [r.get("tag") for r in records]
    return {
        "n": len(records),
        "gap_min": float(gap.min()), "gap_max": float(gap.max()),
        "correlation": correlate(alignment, gap),
        "permutation": permutation_test(alignment, gap),
        "loo_r2": loo_r2(alignment, gap),
        "screen": screen(alignment, gap, tags=tags),
    }
