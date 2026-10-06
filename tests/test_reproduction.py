"""Regression tests: does this code still reproduce the paper?

Run with ``pytest`` from the repository root. Nothing here needs a GPU, a
network connection, or the raw embeddings; the tests work from the canonical
result files in ``data/`` plus short self-contained quantum computations.

The suite is organized by what it protects:

``TestQuantumKernel``       the circuit itself, and the state-overlap shortcut
``TestDiagnostics``         alignment, concentration, geometric difference
``TestPublishedStatistics`` every number quoted in the paper
``TestScreeningRule``       the decision claim, which is what the paper rests on
``TestConfigIntegrity``     the grid definition against the released records
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from qks.analysis import (apply_fitted_model, correlate, fit_model, leave_one_domain_out,
                          leave_one_encoder_out, loo_r2, permutation_test, screen,
                          threshold_sweep)
from qks.config import (ALL_TAGS, GRID, HELDOUT_TAGS, MAIN_TAGS, N_QUBITS,
                        SCREEN_THRESHOLD, metric_for, path)
from qks.diagnostics import (alignment_variants, classical_gram, concentration,
                             geometric_difference, kta, mean_offdiagonal)
from qks.kernels import FEATURE_MAPS, make_fast_kernel, make_pair_kernel

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split():
    """The released main / held-out split."""
    with open(path("grid_split.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def main_grid(split):
    return split["main"]


@pytest.fixture(scope="module")
def heldout(split):
    return split["heldout"]


@pytest.fixture(scope="module")
def main_arrays(main_grid):
    """``(A_C, A_Q, gap)`` for the 18 main-grid configurations."""
    return (np.array([r["c_kta"] for r in main_grid]),
            np.array([r["kta"] for r in main_grid]),
            np.array([r["gap"] for r in main_grid]))


# --------------------------------------------------------------------------


class TestQuantumKernel:
    """The kernel implementation. If these fail nothing else is meaningful."""

    @pytest.mark.parametrize("name", list(FEATURE_MAPS))
    def test_fast_kernel_equals_pairwise_circuit(self, name):
        """The state-overlap route is exact, not an approximation.

        This is the assumption that lets the pipeline compute a kernel in
        seconds instead of 83 minutes, so it is tested for every feature map.
        """
        fmap, n_wires = FEATURE_MAPS[name]
        X = np.random.RandomState(3).uniform(0, np.pi, (10, N_QUBITS))
        pair = make_pair_kernel(fmap, n_wires)
        slow = np.array([[pair(X[i], X[j]) for j in range(len(X))] for i in range(len(X))])
        fast = make_fast_kernel(fmap, n_wires)(X)
        assert np.abs(fast - slow).max() < 1e-10

    @pytest.mark.parametrize("name", list(FEATURE_MAPS))
    def test_kernel_is_a_valid_gram_matrix(self, name):
        """Unit diagonal, symmetric, entries in [0, 1], positive semi-definite."""
        fmap, n_wires = FEATURE_MAPS[name]
        X = np.random.RandomState(11).uniform(0, np.pi, (12, N_QUBITS))
        K = make_fast_kernel(fmap, n_wires)(X)
        assert np.allclose(np.diag(K), 1.0, atol=1e-9)
        assert np.allclose(K, K.T, atol=1e-12)
        assert K.min() >= -1e-12 and K.max() <= 1.0 + 1e-12
        assert np.linalg.eigvalsh(K).min() > -1e-8

    def test_rectangular_kernel_orientation(self):
        """``K(A, B)`` is ``len(A) x len(B)``, the orientation sklearn expects."""
        rng = np.random.RandomState(5)
        A, B = rng.uniform(0, np.pi, (7, N_QUBITS)), rng.uniform(0, np.pi, (4, N_QUBITS))
        assert make_fast_kernel()(A, B).shape == (7, 4)


class TestDiagnostics:
    """Mathematical properties the diagnostics must satisfy."""

    def test_alignment_is_one_for_the_ideal_kernel(self):
        """A kernel that exactly encodes the labels aligns perfectly with them."""
        y = np.array([0, 0, 1, 1, 2, 2])
        ideal = (y[:, None] == y[None, :]).astype(float)
        assert kta(ideal, y) == pytest.approx(1.0, abs=1e-12)

    def test_alignment_is_bounded(self):
        """Alignment of any valid Gram matrix lies in [0, 1]."""
        rng = np.random.RandomState(0)
        X = rng.uniform(0, np.pi, (30, N_QUBITS))
        y = rng.randint(0, 3, 30)
        value = kta(classical_gram(X, "rbf_1/d"), y)
        assert 0.0 <= value <= 1.0

    def test_concentration_bounds(self):
        """The identity gives rho = 1; a constant matrix approaches 1/n."""
        n = 20
        assert concentration(np.eye(n)) == pytest.approx(1.0, abs=1e-9)
        constant = np.ones((n, n)) + 1e-9 * np.eye(n)
        assert concentration(constant) < 2.0 / n

    def test_geometric_difference_increases_as_the_ridge_tightens(self):
        """g grows monotonically as the regularization is loosened.

        Tested on a kernel built from real features rather than random ones:
        the effect comes from the classical kernel being ill-conditioned, which
        random uniform inputs in eight dimensions are not.
        """
        rng = np.random.RandomState(1)
        # two well-separated clusters produce the near-singular classical kernel
        # that makes the ridge matter, as real embeddings do
        X = np.vstack([rng.normal(0.6, 0.05, (20, N_QUBITS)),
                       rng.normal(2.5, 0.05, (20, N_QUBITS))]).clip(0, np.pi)
        K_q = make_fast_kernel()(X)
        K_c = classical_gram(X, "rbf_1/d")
        values = [geometric_difference(K_q, K_c, ridge=r)
                  for r in (1e-4, 1e-5, 1e-6, 1e-7)]
        assert all(b > a for a, b in zip(values, values[1:])), values

    def test_published_lambda_sweep_spans_an_order_of_magnitude(self, main_grid):
        """The paper's point: whether a configuration looks advantaged is set by lambda.

        Every configuration crosses the sqrt(N) boundary somewhere in the sweep,
        so the regime a configuration falls in is decided by the regularization
        rather than by the data.
        """
        boundary = np.sqrt(1000)
        for record in main_grid:
            sweep = record["g_lambda_sweep"]
            loose, tight = sweep["1e-04"], sweep["1e-07"]
            assert tight > 8 * loose, record["tag"]
            assert loose < boundary < tight, record["tag"]

    def test_alignment_variants_cover_all_six_kernels(self):
        rng = np.random.RandomState(2)
        X, y = rng.uniform(0, np.pi, (25, N_QUBITS)), rng.randint(0, 2, 25)
        assert len(alignment_variants(X, y)) == 6


class TestPublishedStatistics:
    """Every number the paper quotes, recomputed from the released records."""

    def test_grid_sizes(self, main_grid, heldout):
        assert len(main_grid) == 18
        assert len(heldout) == 12

    def test_pearson_and_spearman(self, main_arrays):
        A_C, _, gap = main_arrays
        c = correlate(A_C, gap)
        assert c["pearson_r"] == pytest.approx(0.873, abs=5e-4)
        assert c["pearson_p"] < 1e-5
        assert c["spearman_rho"] == pytest.approx(0.775, abs=5e-4)

    def test_quantum_alignment_correlation(self, main_arrays):
        _, A_Q, gap = main_arrays
        assert correlate(A_Q, gap)["pearson_r"] == pytest.approx(0.898, abs=5e-4)

    def test_out_of_sample_r2(self, main_arrays):
        A_C, A_Q, gap = main_arrays
        assert loo_r2(A_C, gap) == pytest.approx(0.709, abs=5e-4)
        assert loo_r2(A_Q, gap) == pytest.approx(0.754, abs=5e-4)

    def test_theory_diagnostics_do_not_reach_significance(self, main_grid, main_arrays):
        """Concentration and the geometric difference fail where alignment works."""
        _, _, gap = main_arrays
        rho = correlate([r["concentration"] for r in main_grid], gap)
        g = correlate([r["g_exact"] for r in main_grid], gap)
        assert rho["pearson_r"] == pytest.approx(-0.153, abs=5e-4)
        assert rho["pearson_p"] > 0.05
        assert g["pearson_r"] == pytest.approx(0.420, abs=5e-4)
        assert g["pearson_p"] > 0.05

    def test_combined_model_is_worse_than_alignment_alone(self, main_grid, main_arrays):
        """Adding the theory diagnostics degrades a model that already works."""
        _, A_Q, gap = main_arrays
        combined = np.column_stack([A_Q,
                                    [r["concentration"] for r in main_grid],
                                    [r["g_exact"] for r in main_grid]])
        assert loo_r2(combined, gap) == pytest.approx(0.697, abs=5e-4)
        assert loo_r2(combined, gap) < loo_r2(A_Q, gap)

    def test_permutation_test(self, main_arrays):
        A_C, _, gap = main_arrays
        assert permutation_test(A_C, gap, n_permutations=2000)["n_exceeding"] == 0

    def test_leave_one_domain_out(self, main_grid, main_arrays):
        A_C, _, gap = main_arrays
        lodo = leave_one_domain_out(A_C, gap, [r["domain"] for r in main_grid])
        assert lodo["text"]["pearson_r"] == pytest.approx(0.932, abs=5e-4)
        assert lodo["vision"]["pearson_r"] == pytest.approx(0.590, abs=5e-4)
        assert lodo["multimodal"]["pearson_r"] == pytest.approx(0.135, abs=5e-4)
        assert lodo["pooled"]["pearson_r"] == pytest.approx(0.828, abs=5e-4)
        assert lodo["pooled"]["mae"] == pytest.approx(0.015, abs=5e-4)

    def test_leave_one_encoder_out_range(self, main_grid, main_arrays):
        A_C, _, gap = main_arrays
        values = [v["pearson_r"] for v in
                  leave_one_encoder_out(A_C, gap, [r["encoder"] for r in main_grid]).values()]
        assert min(values) == pytest.approx(0.845, abs=5e-4)
        assert max(values) == pytest.approx(0.931, abs=5e-4)

    def test_no_quantum_advantage_anywhere(self, main_grid, heldout):
        """The scope claim: the quantum kernel never wins by a meaningful margin."""
        gaps = np.array([r["gap"] for r in main_grid + heldout])
        assert gaps.max() <= 0.013
        assert (gaps[np.array([r["gap"] for r in main_grid + heldout]) > 0.005]).size <= 2


class TestScreeningRule:
    """The decision claim, which is the one the paper rests on."""

    def test_main_grid_screen(self, main_grid, main_arrays):
        A_C, _, gap = main_arrays
        s = screen(A_C, gap, tags=[r["tag"] for r in main_grid])
        assert s["n_flagged"] == 8
        assert s["flagged_mean_gap"] == pytest.approx(-0.044, abs=5e-4)
        assert s["worst_flagged_gap"] == pytest.approx(-0.011, abs=5e-4)
        assert s["all_flagged_are_deficits"] is True

    def test_nothing_flagged_was_worth_running(self, main_arrays):
        """The property a screen must have: no flagged configuration is a win."""
        A_C, _, gap = main_arrays
        assert screen(A_C, gap)["worst_flagged_gap"] <= 0.0

    def test_threshold_is_not_delicate(self, main_arrays):
        """Between 0.02 and 0.15 the worst flagged gap never moves."""
        A_C, _, gap = main_arrays
        worst = [row["worst_flagged_gap"] for row in threshold_sweep(A_C, gap)]
        assert all(w == pytest.approx(-0.011, abs=5e-4) for w in worst)

    def test_heldout_transfer(self, main_arrays, heldout):
        """The model is fitted on the main grid and applied without refitting."""
        A_C, _, gap = main_arrays
        A_held = np.array([r["c_kta"] for r in heldout])
        gap_held = np.array([r["gap"] for r in heldout])

        s = screen(A_held, gap_held)
        assert s["n_flagged"] == 5
        assert s["worst_flagged_gap"] == pytest.approx(0.003, abs=5e-4)

        transfer = apply_fitted_model(fit_model(A_C, gap), A_held, gap_held)
        assert transfer["pearson_r"] == pytest.approx(0.370, abs=5e-3)
        assert transfer["pearson_p"] > 0.05        # not significant, as reported

    def test_heldout_is_a_compressed_regime(self, heldout):
        """Every held-out alignment sits at the low end of the fitted range."""
        assert max(r["c_kta"] for r in heldout) < 0.12


class TestConfigIntegrity:
    """The grid definition must match the released records exactly."""

    def test_tag_counts(self):
        assert len(MAIN_TAGS) == 18
        assert len(HELDOUT_TAGS) == 12
        assert len(set(ALL_TAGS)) == 30

    def test_tags_match_canonical(self):
        with open(path("grid30_canonical.json")) as fh:
            canonical = {r["tag"] for r in json.load(fh)}
        assert canonical == set(ALL_TAGS)

    def test_domain_and_metric_match_canonical(self):
        with open(path("grid30_canonical.json")) as fh:
            canonical = {r["tag"]: r for r in json.load(fh)}
        for tag in ALL_TAGS:
            assert GRID[tag]["domain"] == canonical[tag]["domain"], tag
            assert metric_for(tag) == canonical[tag]["metric"], tag

    def test_encoder_match_canonical(self):
        """``-joint`` is a repository-side distinction, not part of the records."""
        with open(path("grid30_canonical.json")) as fh:
            canonical = {r["tag"]: r for r in json.load(fh)}
        for tag in ALL_TAGS:
            assert GRID[tag]["encoder"].replace("-joint", "") == canonical[tag]["encoder"], tag

    def test_every_domain_is_balanced(self):
        """Three domains of six in the main grid, twelve multilingual held out."""
        from collections import Counter
        counts = Counter(GRID[t]["domain"] for t in MAIN_TAGS)
        assert counts == {"text": 6, "vision": 6, "multimodal": 6}
        assert all(GRID[t]["domain"] == "multilingual" for t in HELDOUT_TAGS)


@pytest.mark.skipif(not os.path.exists(path("featuremap_ablation_all30.json")),
                    reason="ablation results not present")
class TestAblation:
    """The kernel-robustness result, if the ablation file is available."""

    @staticmethod
    @pytest.fixture(scope="class")
    def ablation():
        with open(path("featuremap_ablation_all30.json")) as fh:
            return json.load(fh)

    def test_covers_all_configurations_and_kernels(self, ablation):
        assert len(ablation) == 30
        for record in ablation.values():
            assert set(record["kernels"]) == set(FEATURE_MAPS)

    def test_paper_kernel_column_reproduces_the_main_run(self, ablation, main_grid):
        """The IQP-1L column must equal the published gap for every configuration."""
        canonical = {r["tag"]: r["gap"] for r in main_grid}
        for tag, gap in canonical.items():
            assert ablation[tag]["kernels"]["IQP-1L"]["gap"] == pytest.approx(gap, abs=1e-6)

    def test_screen_holds_under_every_eight_qubit_kernel(self, ablation, main_grid):
        """At eight qubits, every flagged configuration is a deficit under every circuit."""
        tags = [r["tag"] for r in main_grid]
        flagged = [t for t in tags if ablation[t]["c_kta"] < SCREEN_THRESHOLD]
        for name in ("IQP-1L", "IQP-2L", "Angle", "IQP-1L-all"):
            worst = max(ablation[t]["kernels"][name]["gap"] for t in flagged)
            assert worst <= 0.0, f"{name} leaks: worst flagged gap {worst:+.3f}"

    def test_reduced_qubit_counts_are_where_the_rule_stops(self, ablation, main_grid):
        """At six and four qubits one flagged configuration crosses parity."""
        tags = [r["tag"] for r in main_grid]
        flagged = [t for t in tags if ablation[t]["c_kta"] < SCREEN_THRESHOLD]
        for name in ("IQP-1L-6q", "IQP-1L-4q"):
            assert max(ablation[t]["kernels"][name]["gap"] for t in flagged) > 0.0
