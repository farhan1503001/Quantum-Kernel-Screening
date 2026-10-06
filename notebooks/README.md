# Original notebooks

The 76 Colab notebooks the study was actually run in, kept for provenance. The
`qks/` package is a cleaned reimplementation of the final pipeline these
converged on, verified to reproduce their output exactly; see
`scripts/00_verify_install.py`.

Use the package for anything new. Use these to check what the original run did.

## How the package maps onto them

| package module | derived from |
|---|---|
| `qks/encoders.py` | `pipeline_final/seeds_A_embeddings.ipynb`, plus the per-encoder `week2_*` notebooks |
| `qks/datasets.py` | the loader cells of `pipeline_final/seeds_A_embeddings.ipynb` and `week2_*` |
| `qks/features.py` | the PCA and scaling cells, identical across every `week2_*` and `week3_*` notebook |
| `qks/kernels.py` | `pipeline_final/seeds_B_kernels.ipynb` and `featuremap_ablation_all30.ipynb` |
| `qks/diagnostics.py` | `pipeline_final/recompute_diagnostics.ipynb` |
| `qks/evaluate.py` | the evaluation cells of `pipeline_final/seeds_B_kernels.ipynb` |
| `qks/analysis.py` | `week5_6_analysis/week5_diagnostic_analysis_DRIVE.ipynb` |

## The folders

**`pipeline_final/`** — the five notebooks that define the published pipeline.
These are the authoritative ones and the package follows them line for line.

| notebook | what it does |
|---|---|
| `seeds_A_embeddings.ipynb` | encoders and dataset loaders, writes raw embeddings |
| `seeds_B_kernels.ipynb` | quantum kernels, baselines, the gap, the seed study |
| `featuremap_ablation_all30.ipynb` | all 30 configurations under six quantum kernels |
| `recompute_diagnostics.ipynb` | the lambda sweep and the off-diagonal statistics |
| `recompute_kernel_choice_curves.ipynb` | alignment under six classical kernels, learning curves |

**`week2_embeddings/`** — embedding extraction, one notebook per configuration.
This is where each encoder's representation convention was fixed.

**`week3_kernels/`** — quantum kernel computation per configuration, using the
original pairwise route at roughly 83 minutes per kernel matrix.

**`week4_ablations/`** — the first feature-map and qubit-count ablations, on the
24 configurations available at the time. Superseded by
`pipeline_final/featuremap_ablation_all30.ipynb`, which covers all 30 and agrees
with these to 5e-7.

**`week5_6_analysis/`** — diagnostic analysis and the robustness hardening pass.

**`per_config/`** — later additions and completions that do not fit the weekly
numbering, including the FLAVA and ViT configurations added after the main grid
was first assembled.

## Two cautions

Many notebooks are superseded versions of the same configuration. Where a
`_DRIVE` or `_autosave` variant exists it is the later one. Where both a weekly
notebook and a `pipeline_final/` notebook cover the same ground, the latter is
authoritative.

A regularization sweep appears in the `week4_*` notebooks. It ran on a
200-point subset rather than the full 1000, so it is not comparable with the
main results and is deliberately not reported in the paper or reimplemented in
the package.
