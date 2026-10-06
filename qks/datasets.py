"""Dataset loaders, reproducing the sampling of the original run exactly.

Each loader returns ``(train_items, y_train, test_items, y_test)`` where the
items are raw texts, PIL images, or ``(image, text)`` pairs depending on the
task. Embedding extraction happens in ``scripts/01_extract_embeddings.py``.

How the seed is used differs by configuration, and the difference is reported in
the paper because it affects how seed variability should be read:

==================  =======================================================
loader              what the seed controls
==================  =======================================================
``load_text``       the pool draw (``shuffle(seed)``), so new examples
``load_fashion``    the pool draw (``RandomState(seed)``), so new examples
``load_cifar10``    only the shuffle of a pool fixed by dataset order
``load_mscoco``     only the mismatched-caption pairing
``load_snlive``     nothing; the pool is fixed by dataset order
``load_multilingual`` the pool draw within one language
==================  =======================================================

For the three configurations whose pool is fixed by dataset order, seeds
resample within that pool rather than drawing new examples, so their seed spread
understates full resampling variance.
"""

from __future__ import annotations

import random

import numpy as np

from .config import DATASETS, LANGUAGES


def _hf(hub: str, config: str | None = None, split: str | None = None, **kw):
    from datasets import load_dataset
    args = (hub, config) if config else (hub,)
    return load_dataset(*args, split=split, **kw) if split else load_dataset(*args, **kw)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def load_text(dataset: str, seed: int = 42):
    """SST-2 or AG News. Pool is 1100 train / 250 test drawn by seeded shuffle."""
    spec = DATASETS[dataset]
    n_train, n_test = spec["pool"]
    ds = _hf(spec["hub"], spec["config"])
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    test = ds[spec["test_split"]].shuffle(seed=seed).select(range(n_test))
    col = spec["text"]
    return ([x[col] for x in train], np.array(train["label"]),
            [x[col] for x in test], np.array(test["label"]))


def load_multilingual(language: str, seed: int = 42):
    """One language of the three-class sentiment corpus.

    The six language subsets are independently sourced texts, not translations
    of one another, so cross-lingual comparability rests on the shared task.
    """
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; expected one of {LANGUAGES}")
    spec = DATASETS["sentiment"]
    n_train, n_test = spec["pool"]
    ds = _hf(spec["hub"], language)
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    test = ds["test"].shuffle(seed=seed).select(range(n_test))
    col = spec["text"]
    return ([x[col] for x in train], np.array(train["label"]),
            [x[col] for x in test], np.array(test["label"]))


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------


def _balanced_indices(labels, class_a, class_b, n_total, seed):
    """Balanced draw of ``n_total`` indices over two classes, shuffled."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    idx_a = np.where(labels == class_a)[0]
    rng.shuffle(idx_a)
    idx_b = np.where(labels == class_b)[0]
    rng.shuffle(idx_b)
    half = n_total // 2
    idx = np.concatenate([idx_a[:half], idx_b[:half]])
    rng.shuffle(idx)
    return idx


def load_cifar10(seed: int = 42):
    """CIFAR-10 airplane vs automobile.

    The candidate pool is the first 550 images of each class in dataset order,
    which is fixed. The seed only shuffles that pool, so every seed sees the
    same examples in a different order.
    """
    spec = DATASETS["cifar10"]
    ds = _hf(spec["hub"])
    col = "img" if "img" in ds["train"].column_names else "image"
    a, b = spec["class_ids"]
    out = {}
    for split, keep_per_class in (("train", 550), ("test", 125)):
        images, labels = [], []
        for sample in ds[split]:
            if sample["label"] in (a, b):
                images.append(sample[col])
                labels.append(sample["label"])
            if labels.count(a) >= 550 and labels.count(b) >= 550:
                break
        first_a = [i for i, l in enumerate(labels) if l == a][:keep_per_class]
        first_b = [i for i, l in enumerate(labels) if l == b][:keep_per_class]
        order = first_a + first_b
        random.seed(seed)
        perm = list(range(len(order)))
        random.shuffle(perm)
        out[split] = ([images[order[i]] for i in perm],
                      np.array([labels[order[i]] for i in perm]))
    return out["train"][0], out["train"][1], out["test"][0], out["test"][1]


def load_fashion(seed: int = 42):
    """Fashion-MNIST T-shirt vs trouser. The seed draws the pool, already 1000/200."""
    spec = DATASETS["fashion"]
    ds = _hf(spec["hub"])
    col = "image" if "image" in ds["train"].column_names else "img"
    a, b = spec["class_ids"]
    n_train, n_test = spec["pool"]
    train = ds["train"].select(_balanced_indices(ds["train"]["label"], a, b, n_train, seed))
    test = ds["test"].select(_balanced_indices(ds["test"]["label"], a, b, n_test, seed))
    return train[col], np.array(train["label"]), test[col], np.array(test["label"])


# --------------------------------------------------------------------------
# Multimodal
# --------------------------------------------------------------------------


def load_mscoco(seed: int = 42, n_pool: int = 780, n_train_images: int = 600,
                n_test_images: int = 130):
    """MSCOCO image-caption matching.

    Each image contributes two pairs, one with its own caption (label 1) and one
    with a caption drawn from another image (label 0), so 600 training images
    give 1200 training pairs. The image pool is fixed by dataset order; the seed
    only decides which caption each image is mismatched with.

    Returns ``(train_pairs, y_train, test_pairs, y_test)`` where each pair is
    ``(image, caption)``.
    """
    from PIL import Image

    spec = DATASETS["mscoco"]
    images, captions = [], []
    try:
        ds = _hf(spec["hub"], split="test")
        cols = ds.column_names
        img_col = "image" if "image" in cols else next(c for c in cols if "image" in c.lower())
        cap_col = ("caption" if "caption" in cols else
                   "captions" if "captions" in cols else
                   next(c for c in cols if "caption" in c.lower()))
        for sample in ds:
            caption = sample[cap_col]
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            image = sample[img_col]
            if not isinstance(image, Image.Image):
                continue
            images.append(image.convert("RGB"))
            captions.append(str(caption))
            if len(images) >= n_pool:
                break
    except Exception as exc:                                   # pragma: no cover
        print(f"  primary MSCOCO source failed ({type(exc).__name__}); using lmms-lab fallback")
        for sample in _hf("lmms-lab/COCO-Caption", split="val", streaming=True):
            image, caption = sample.get("image"), sample.get("answer")
            if isinstance(caption, list) and caption:
                caption = caption[0]
            if not isinstance(image, Image.Image):
                continue
            images.append(image.convert("RGB"))
            captions.append(str(caption))
            if len(images) >= n_pool:
                break

    needed = n_train_images + n_test_images
    assert len(images) >= needed, f"MSCOCO: got {len(images)} images, need {needed}"

    rng = np.random.RandomState(seed)

    def build(index_range):
        pairs = []
        for i in index_range:
            pairs.append((i, i, 1))
            j = rng.choice([k for k in index_range if k != i])
            pairs.append((i, int(j), 0))
        rng.shuffle(pairs)
        return pairs

    train_spec = build(list(range(n_train_images)))
    test_spec = build(list(range(n_train_images, needed)))
    assemble = lambda spec_: ([(images[i], captions[c]) for i, c, _ in spec_],
                              np.array([lab for _, _, lab in spec_]))
    (train_pairs, y_train), (test_pairs, y_test) = assemble(train_spec), assemble(test_spec)
    return train_pairs, y_train, test_pairs, y_test


def load_snlive(seed: int = 42, n_train_per_class: int = 400, n_test_per_class: int = 90):
    """SNLI-VE three-class visual entailment.

    The pool is the first ``n`` examples of each class in dataset order, so it
    does not depend on the seed at all. ``seed`` is accepted for interface
    symmetry and deliberately unused.

    Returns ``(train_pairs, y_train, test_pairs, y_test)`` with
    ``(image, hypothesis)`` pairs.
    """
    from PIL import Image

    spec = DATASETS["snlive"]
    ds = _hf(spec["hub"])
    cols = ds["train"].column_names
    img_col = "image" if "image" in cols else next(
        c for c in cols if "image" in c.lower() or "img" in c.lower())
    txt_col = "hypothesis" if "hypothesis" in cols else next(
        c for c in cols if "hypo" in c.lower() or "sentence" in c.lower())
    lbl_col = "label" if "label" in cols else next(
        c for c in cols if "label" in c.lower() or "gold" in c.lower())
    str_to_int = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def label_of(sample):
        value = sample[lbl_col]
        return str_to_int.get(value, value) if isinstance(value, str) else value

    def balanced(split, n_per_class):
        pairs, labels, counts = [], [], {0: 0, 1: 0, 2: 0}
        for sample in ds[split]:
            label = label_of(sample)
            if label not in (0, 1, 2) or counts[label] >= n_per_class:
                continue
            image = sample[img_col]
            if not isinstance(image, Image.Image):
                continue
            pairs.append((image.convert("RGB"), sample[txt_col]))
            labels.append(label)
            counts[label] += 1
            if all(v >= n_per_class for v in counts.values()):
                break
        return pairs, np.array(labels)

    test_split = "test" if "test" in ds else "validation"
    train_pairs, y_train = balanced("train", n_train_per_class)
    test_pairs, y_test = balanced(test_split, n_test_per_class)
    return train_pairs, y_train, test_pairs, y_test


#: Dispatch from a dataset name to its loader.
LOADERS = {
    "sst2": lambda seed=42: load_text("sst2", seed),
    "agnews": lambda seed=42: load_text("agnews", seed),
    "cifar10": load_cifar10,
    "fashion": load_fashion,
    "mscoco": load_mscoco,
    "snlive": load_snlive,
    "sentiment": load_multilingual,
}
