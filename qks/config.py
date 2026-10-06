"""Experiment configuration: the grid, the fixed pipeline constants, and paths.

Everything in this module is a constant of the study. No value here is tuned or
searched over; the robustness analyses vary these explicitly and are implemented
in ``scripts/04_ablation_featuremaps.py`` and ``scripts/06_analysis.py``.

The grid has two parts, kept separate by design:

* ``MAIN_GRID`` (18 configurations) -- six datasets in three domains, each paired
  with three encoders. All model fitting and all headline statistics come from
  this block.
* ``HELDOUT_GRID`` (12 configurations) -- three-class sentiment in six languages
  under two encoders. No fitted model in the paper uses it. The screening
  threshold was fixed during development on a set that included it, so results
  on this block are a consistency check rather than a strict held-out test.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Fixed pipeline constants. These are identical for every configuration.
# --------------------------------------------------------------------------

N_QUBITS = 8          # one qubit per PCA component
N_TRAIN = 1000        # training points per configuration
N_TEST = 200          # test points per configuration
SEED = 42             # the published seed
C_SVM = 1.0           # SVM regularization, shared by quantum and classical
FEATURE_RANGE = (0.0, 3.141592653589793)   # min-max target range [0, pi]
RBF_GAMMA_DIAGNOSTIC = "1/d"               # bandwidth for the alignment RBF kernel
N_BOOTSTRAP = 1000                         # resamples for the gap interval
N_PERMUTATION = 10_000                     # shuffles for the permutation test
SCREEN_THRESHOLD = 0.05                    # the screening rule: skip if A_C < this

# Seeds used by the seed-variability study (scripts/05_seed_study.py).
SEEDS_STUDY = (42, 7, 123)

# --------------------------------------------------------------------------
# Encoders. ``repr`` records what is extracted BEFORE PCA, which differs by
# family and is the single most error-prone part of the pipeline to reproduce.
# --------------------------------------------------------------------------

ENCODERS = {
    # masked-LM family: first-token final-layer state, NOT normalized
    "BERT":    dict(ckpt="bert-base-uncased",                 dim=768,  repr="first_token", norm=False),
    "RoBERTa": dict(ckpt="roberta-base",                      dim=768,  repr="first_token", norm=False),
    "mBERT":   dict(ckpt="bert-base-multilingual-cased",      dim=768,  repr="first_token", norm=False),
    "XLM-R":   dict(ckpt="xlm-roberta-base",                  dim=768,  repr="first_token", norm=False),

    # sentence encoder: sentence-transformers does mean pooling AND L2
    # normalization inside .encode(). This differs from the rest of the
    # masked-LM family -- see README, "Known deviations".
    "MiniLM":  dict(ckpt="sentence-transformers/all-MiniLM-L6-v2", dim=384,
                    repr="mean_pool_st", norm=True),

    # vision encoders, all L2-normalized
    "CLIP":    dict(ckpt="openai/clip-vit-base-patch32",      dim=512,  repr="visual_projection", norm=True),
    "DINOv2":  dict(ckpt="facebook/dinov2-base",              dim=768,  repr="cls_token",         norm=True),
    "ViT":     dict(ckpt="google/vit-base-patch16-224",       dim=768,  repr="cls_token",         norm=True),

    # joint multimodal encoders: image and text vectors concatenated
    "CLIP-joint":   dict(ckpt="openai/clip-vit-base-patch32",       dim=1024, repr="proj_img+txt",  norm=True),
    "SigLIP-joint": dict(ckpt="google/siglip-base-patch16-224",     dim=1536, repr="pooled_img+txt", norm=True),
    "FLAVA-joint":  dict(ckpt="facebook/flava-full",                dim=1536, repr="cls_img+txt",    norm=True),
}

# --------------------------------------------------------------------------
# Datasets. ``pool`` is the balanced pool built before the fixed-seed draw of
# N_TRAIN / N_TEST. ``metric`` selects the F1 averaging rule.
# --------------------------------------------------------------------------

DATASETS = {
    "sst2":      dict(hub="nyu-mll/glue", config="sst2", text="sentence", test_split="validation",
                      classes=2, metric="binary", pool=(1100, 250), task="Sentiment"),
    "agnews":    dict(hub="fancyzhx/ag_news", config=None, text="text", test_split="test",
                      classes=4, metric="macro", pool=(1100, 250), task="Topic"),
    "cifar10":   dict(hub="uoft-cs/cifar10", config=None, classes=2, metric="binary",
                      pool=(1100, 250), task="Airplane vs automobile", class_ids=(0, 1)),
    "fashion":   dict(hub="zalando-datasets/fashion_mnist", config=None, classes=2, metric="binary",
                      pool=(1000, 200), task="T-shirt vs trouser", class_ids=(0, 1)),
    "mscoco":    dict(hub="nlphuji/mscoco_2014_5k_test_image_text_retrieval", config=None,
                      classes=2, metric="binary", pool=(1200, 260), task="Image-caption matching"),
    "snlive":    dict(hub="sedrickkeh/snli-ve", config=None, classes=3, metric="macro",
                      pool=(1200, 270), task="Visual entailment"),
    "sentiment": dict(hub="tyqiangz/multilingual-sentiments", config=None, text="text",
                      classes=3, metric="macro", pool=(1000, 200), task="Sentiment, 3-class"),
}

LANGUAGES = ("arabic", "chinese", "english", "german", "hindi", "spanish")

# --------------------------------------------------------------------------
# The grid. ``tag`` is the identifier used in every filename and every record.
# Tags are historical and deliberately left unchanged so that the released
# artifacts match the filenames in the canonical JSON files.
# --------------------------------------------------------------------------

MAIN_GRID = [
    # tag,               dataset,    encoder,        domain
    ("sst2_bert",        "sst2",     "BERT",         "text"),
    ("ag_news",          "agnews",   "BERT",         "text"),
    ("sst2_minilm",      "sst2",     "MiniLM",       "text"),
    ("agnews_minilm",    "agnews",   "MiniLM",       "text"),
    ("sst2_roberta",     "sst2",     "RoBERTa",      "text"),
    ("agnews_roberta",   "agnews",   "RoBERTa",      "text"),

    ("cifar10",          "cifar10",  "CLIP",         "vision"),
    ("fashion_clip",     "fashion",  "CLIP",         "vision"),
    ("cifar10_dinov2",   "cifar10",  "DINOv2",       "vision"),
    ("fashion_dinov2",   "fashion",  "DINOv2",       "vision"),
    ("cifar10_vit",      "cifar10",  "ViT",          "vision"),
    ("fashion_vit",      "fashion",  "ViT",          "vision"),

    ("mscoco",           "mscoco",   "CLIP-joint",   "multimodal"),
    ("snlive",           "snlive",   "CLIP-joint",   "multimodal"),
    ("mscoco_siglip",    "mscoco",   "SigLIP-joint", "multimodal"),
    ("snlive_siglip",    "snlive",   "SigLIP-joint", "multimodal"),
    ("mscoco_flava",     "mscoco",   "FLAVA-joint",  "multimodal"),
    ("snlive_flava",     "snlive",   "FLAVA-joint",  "multimodal"),
]

HELDOUT_GRID = [
    (f"multiling_{enc_tag}_{lang}", "sentiment", enc, "multilingual", lang)
    for enc, enc_tag in (("mBERT", "mbert"), ("XLM-R", "xlmr"))
    for lang in LANGUAGES
]

# Convenience lookups -------------------------------------------------------

GRID = {t: dict(dataset=d, encoder=e, domain=dom) for t, d, e, dom in MAIN_GRID}
GRID.update({t: dict(dataset=d, encoder=e, domain=dom, language=lang)
             for t, d, e, dom, lang in HELDOUT_GRID})

MAIN_TAGS = [t for t, *_ in MAIN_GRID]
HELDOUT_TAGS = [t for t, *_ in HELDOUT_GRID]
ALL_TAGS = MAIN_TAGS + HELDOUT_TAGS

DOMAIN_COLORS = {          # the palette used by every figure and table
    "text": "#0072B2",
    "vision": "#E69F00",
    "multimodal": "#009E73",
    "multilingual": "#CC79A7",
}


def metric_for(tag: str) -> str:
    """Return 'binary' or 'macro' for a configuration tag."""
    return DATASETS[GRID[tag]["dataset"]]["metric"]


# --------------------------------------------------------------------------
# Paths. Override with the QKS_DATA_DIR environment variable.
# --------------------------------------------------------------------------

DATA_DIR = os.environ.get("QKS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data"))


def path(name: str) -> str:
    """Absolute path to a file inside the data directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


if __name__ == "__main__":
    print(f"main grid     : {len(MAIN_TAGS)} configurations")
    print(f"held-out block: {len(HELDOUT_TAGS)} configurations")
    print(f"total         : {len(ALL_TAGS)} configurations")
    print(f"encoders      : {len(ENCODERS)}")
    print(f"data dir      : {DATA_DIR}")
