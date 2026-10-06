# Before the First Circuit

Code and data for *Zero-Shot Quantum Kernel Selection on Foundation Models with
Kernel-Target Alignment*.

Computing a quantum kernel is expensive, and on most encoder and task pairs it
does not beat a classical alternative. This repository implements the screening
rule from the paper: a classical quantity, computable in under a second from the
training features and labels alone, that predicts the quantum versus classical
performance gap before any circuit is run.

The rule is one line:

```python
from qks.diagnostics import classical_alignment

if classical_alignment(X_train, y_train) < 0.05:
    pass    # skip the quantum kernel; it will not help on this configuration
```

Across the paper's eighteen-configuration main grid this flags eight
configurations, every one of them a deficit, saving about eleven hours of
simulation without passing over a single case where the quantum kernel matched
the best classical kernel.

---

## Quick start

```bash
git clone <repository-url> && cd qkernel-screening
pip install -r requirements.txt
python scripts/00_verify_install.py
```

`00_verify_install.py` needs no GPU, no network, and no downloaded embeddings.
It recomputes every statistic the paper reports from the canonical result files
in `data/` and prints a pass or fail line for each. A correct install ends with:

```
ALL 17 CHECKS PASSED. This installation reproduces the paper.
```

To see the full analysis rather than a pass/fail summary:

```bash
python scripts/05_analysis.py
```

---

## What is in here

```
qks/                        the package
  config.py                 the 30-configuration grid, constants, colour palette
  datasets.py               seven dataset loaders, reproducing the original sampling
  encoders.py               eleven frozen encoders, with the library-version guards
  features.py               PCA to 8 dimensions, scaling to [0, pi], the fixed draw
  kernels.py                four quantum feature maps, six kernel settings
  diagnostics.py            alignment, concentration, geometric difference
  evaluate.py               classical baselines, the utility gap, bootstrap intervals
  analysis.py               correlations, permutation test, transfer, the screen
  io.py                     reading and writing artifacts

scripts/
  00_verify_install.py      reproduce every published number (start here)
  01_extract_embeddings.py  stage 1, needs a GPU and network
  02_compute_kernels.py     stage 2, kernels and per-configuration records
  03_ablation_featuremaps.py  the gap under six quantum kernels
  04_seed_study.py          seed variability at the full sample size
  05_analysis.py            the full analysis report

figures/                    one script per paper figure, all regenerate from data/
tests/                      43 regression tests; run with pytest
data/                       canonical result files, tracked in git
notebooks/                  the original exploratory notebooks
```

---

## Reproducing the paper

The pipeline is four stages. Only the first needs a GPU, and the released
canonical files let you skip it entirely if you only want the analysis.

### Stage 1, embeddings (GPU, about 30 to 60 minutes)

```bash
pip install -r requirements-extract.txt
python scripts/01_extract_embeddings.py
```

Downloads each dataset, runs the frozen encoder, and writes
`data/<tag>_seed42_raw.npz`. Configurations that already have a file are
skipped, so an interrupted run can simply be restarted.

### Stage 2, kernels and records (CPU, a few minutes)

```bash
python scripts/02_compute_kernels.py
```

Applies the fixed preprocessing, builds the quantum kernel, fits the three
classical baselines and the quantum SVM, and writes one record per
configuration with the gap, its bootstrap interval, and every diagnostic.

### Stage 3, robustness

```bash
python scripts/03_ablation_featuremaps.py     # six quantum kernels, ~15 min
python scripts/04_seed_study.py               # needs stage 1 with --seeds 42 7 123
```

### Stage 4, analysis and figures

```bash
python scripts/05_analysis.py --from-records  # your own run
python scripts/05_analysis.py                 # the released canonical files
for f in figures/*.py; do python "$f"; done
```

---

## The experimental grid

Thirty configurations, kept in two parts by design.

**Main grid, 18 configurations.** Six datasets in three domains, each paired
with three encoders drawn from distinct pretraining paradigms. All model
fitting and all headline statistics come from this block.

| Domain | Datasets | Encoders |
|---|---|---|
| Text | SST-2, AG News | BERT, RoBERTa, MiniLM |
| Vision | CIFAR-10, Fashion-MNIST | CLIP, DINOv2, ViT |
| Multimodal | MSCOCO, SNLI-VE | CLIP, SigLIP, FLAVA (joint) |

**Held-out block, 12 configurations.** Three-class sentiment in six languages
under mBERT and XLM-R. No fitted model in the paper uses it.

One qualification is worth stating plainly, because it changes how the held-out
evidence should be read. The screening threshold of 0.05 was fixed during
development on a set of configurations that included these twelve, and was not
revised afterwards. The fitted model is genuinely out of sample on this block;
the threshold is not. Results there are a consistency check rather than a strict
held-out test.

---

## The fixed pipeline

Every configuration goes through exactly the same path, which is what makes the
grid comparable:

1. A frozen encoder produces embeddings; weights are never updated.
2. PCA to 8 dimensions, one component per qubit, fitted on the training pool.
3. Min-max scaling to `[0, pi]`, fitted on the training pool, test values
   clipped to the same interval.
4. A fixed-seed draw of 1000 training and 200 test points.
5. A single-layer IQP map on 8 qubits with a linear chain of ZZ interactions;
   the kernel is the fidelity overlap, computed exactly on a statevector
   simulator.
6. Three classical SVMs on the identical features, and a quantum SVM on the
   precomputed Gram matrix, all at `C = 1`.

The utility gap is

```
Delta = F1_quantum - max(F1_linear, F1_rbf, F1_poly)
```

Taking the maximum over three baselines is deliberately conservative: the
quantum kernel has to beat the strongest classical competitor, not an average
one.

### Why the kernel computes in seconds here

The main run evaluated one circuit per pair of points, which is how the kernel
must be obtained on hardware, where states cannot be read out and each overlap
is estimated by measurement. That cost about 83 minutes per kernel matrix at
N = 1000.

Because `K(x, x') = |<phi(x')|phi(x)>|^2` with `|phi(x)> = U(x)|0>`, this code
simulates each point's state once and forms every entry with one matrix product.
On an exact statevector simulator the two routes are *identical*, not
approximately equal. `qks.kernels.self_check()` asserts the equality for every
feature map and is called by every script before anything expensive runs; the
test suite checks it too.

The 83-minute figure in the paper is therefore the hardware-equivalent cost, and
the seconds here are a simulator shortcut, not a different method.

---

## Known deviations

Two things in this repository differ from a naive reading of the paper. Both are
documented here rather than silently fixed, because either would change results
if handled differently.

**MiniLM uses mean pooling, not the first token.** Section 3.3 of the paper
groups MiniLM with BERT, RoBERTa, mBERT and XLM-R and says the first-token
representation is taken without normalization. The original run accessed MiniLM
through `sentence-transformers`, whose `.encode()` applies mean pooling and L2
normalization internally. This code reproduces the original run, so MiniLM is
mean-pooled and normalized. Two configurations are affected, SST-2/MiniLM and
AG News/MiniLM. No reported number changes; the paper's sentence is what is
inaccurate.

**Library versions move the features slightly.** Re-deriving features from raw
embeddings on a different scikit-learn version reproduces the published features
to about 1e-5 rather than exactly, because the randomized PCA solver differs
between releases. The effect on reported quantities:

| quantity | effect |
|---|---|
| F1 scores, utility gap, bootstrap intervals | none; reproduce exactly |
| alignment, concentration, off-diagonals | about 1e-7 |
| geometric difference | about 3e-3, amplified by its ridge inverse |

Two library settings also matter enough that `qks.encoders.prepare_torch()`
forces them, after both were found to move results during the seed study. TF32
must be off, because Ampere GPUs enable it by default and it perturbs the
patch-embedding layer at the 1e-3 level. Image processors must be loaded with
`use_fast=False`, because newer `transformers` resizes with torch rather than
PIL, and for 28x28 and 32x32 inputs upsampled to 224 the two disagree enough to
move the gap.

---

## Data files

`data/` ships the canonical result files, which are tracked in git because every
number in the paper traces to them.

| file | contents |
|---|---|
| `grid30_canonical.json` | all 30 configurations, every field |
| `grid_split.json` | the same records split into `main` and `heldout` |
| `featuremap_ablation_all30.json` | 30 configurations under 6 quantum kernels |

Generated intermediates, raw embeddings, cached kernels and per-run records, are
gitignored. They are large and fully reproducible from the above.

---

## Tests

```bash
pip install pytest
pytest
```

43 tests covering the kernel implementation, the mathematical properties of each
diagnostic, every published statistic, the screening rule, and the integrity of
the grid definition against the released records. They run in about six seconds
and need no GPU or network.

---

## Citation

```bibtex
@article{soumik2026beforethefirstcircuit,
  title  = {Before the First Circuit: Zero-Shot Quantum Kernel Selection on
            Foundation Models with Kernel-Target Alignment},
  author = {Soumik, Mohd. Farhan Israk and Shahid, Abdur R.},
  year   = {2026},
  note   = {Under review}
}
```

## License

See `LICENSE`. The datasets and pretrained encoders carry their own licenses and
are not redistributed here; the loaders fetch them from their original sources.
