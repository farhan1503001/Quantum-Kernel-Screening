"""Frozen encoders. Weights are never updated anywhere in this study.

What each family extracts, and why it matters: the representation taken before
PCA is the most error-prone part of the pipeline to reproduce, because the
convention differs by family and a wrong choice silently changes every
downstream number.

===================  =========================================  ==========
family               representation                             normalized
===================  =========================================  ==========
BERT, RoBERTa,       final-layer state of the first token       no
mBERT, XLM-R
MiniLM               mean pooling via sentence-transformers     yes
CLIP (vision)        pooled vision tower -> visual_projection   yes (L2)
DINOv2, ViT          class token of the final layer             yes (L2)
CLIP/SigLIP/FLAVA    image and text vectors concatenated        yes (L2)
(joint)
===================  =========================================  ==========

Two library-compatibility notes, both of which changed results when they were
missed during the seed study:

1. ``use_fast=False`` on every image processor. Newer ``transformers`` defaults
   to torch-based fast processors that resize differently from the PIL path the
   original run used. For 28x28 and 32x32 inputs upsampled to 224 the two
   disagree enough to move the utility gap.
2. TF32 must be off (see ``disable_tf32``). Ampere GPUs enable TF32
   convolutions by default, which perturbs the patch-embedding layer at the
   1e-3 level. Text encoders have no convolutions and are unaffected.

Both are applied by ``prepare_torch()``, which every extraction script calls
first.
"""

from __future__ import annotations

import numpy as np

_cache: dict = {}
_DEVICE = None


def prepare_torch(verbose: bool = True):
    """Select the device and restore the original run's numerical settings.

    Call this once before any extraction. Returns the device string.
    """
    global _DEVICE
    import torch

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        import transformers
        print(f"device {_DEVICE} | torch {torch.__version__} | "
              f"transformers {transformers.__version__} | TF32 off")
        if _DEVICE == "cpu":
            print("WARNING: no GPU found. Embedding extraction will be very slow.")
    return _DEVICE


def _device():
    return _DEVICE if _DEVICE is not None else prepare_torch(verbose=False)


def _l2(h):
    """Row-wise L2 normalization of a torch tensor."""
    import torch
    return h / torch.linalg.norm(h, dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# Text encoders
# --------------------------------------------------------------------------


def masked_lm_first_token(checkpoint: str, texts, batch: int = 32) -> np.ndarray:
    """First-token final-layer state, unnormalized. BERT, RoBERTa, mBERT, XLM-R."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if checkpoint not in _cache:
        _cache[checkpoint] = (AutoTokenizer.from_pretrained(checkpoint),
                              AutoModel.from_pretrained(checkpoint).to(_device()).eval())
    tok, model = _cache[checkpoint]
    out = []
    for i in range(0, len(texts), batch):
        inputs = tok(list(texts[i:i + batch]), padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to(_device())
        with torch.no_grad():
            out.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(out)


def minilm_sentence(texts, checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
                    batch: int = 64) -> np.ndarray:
    """MiniLM through sentence-transformers.

    Note that ``SentenceTransformer.encode`` applies mean pooling and L2
    normalization internally, so MiniLM does NOT follow the first-token,
    unnormalized convention of the other masked-LM encoders. This matches the
    original run; see README, "Known deviations".
    """
    from sentence_transformers import SentenceTransformer

    key = f"st::{checkpoint}"
    if key not in _cache:
        _cache[key] = SentenceTransformer(checkpoint, device=_device())
    return _cache[key].encode(list(texts), batch_size=batch,
                              show_progress_bar=False, convert_to_numpy=True)


# --------------------------------------------------------------------------
# Vision encoders
# --------------------------------------------------------------------------


def clip_images(images, checkpoint: str = "openai/clip-vit-base-patch32",
                batch: int = 64) -> np.ndarray:
    """CLIP vision tower, pooled output through the visual projection, L2-normalized.

    Computed from the model's parts rather than ``get_image_features`` because
    recent ``transformers`` releases changed that method to return an output
    object. The values are what the original call returned.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if "clip" not in _cache:
        _cache["clip"] = (CLIPProcessor.from_pretrained(checkpoint, use_fast=False),
                          CLIPModel.from_pretrained(checkpoint).to(_device()).eval())
    proc, model = _cache["clip"]
    out = []
    for i in range(0, len(images), batch):
        inputs = proc(images=list(images[i:i + batch]), return_tensors="pt",
                      padding=True).to(_device())
        with torch.no_grad():
            pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            out.append(_l2(model.visual_projection(pooled)).cpu().numpy())
    return np.vstack(out)


def dinov2_images(images, checkpoint: str = "facebook/dinov2-base",
                  batch: int = 64) -> np.ndarray:
    """DINOv2 class token, L2-normalized."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if "dinov2" not in _cache:
        _cache["dinov2"] = (AutoImageProcessor.from_pretrained(checkpoint, use_fast=False),
                            AutoModel.from_pretrained(checkpoint).to(_device()).eval())
    proc, model = _cache["dinov2"]
    out = []
    for i in range(0, len(images), batch):
        inputs = proc(images=[im.convert("RGB") for im in images[i:i + batch]],
                      return_tensors="pt").to(_device())
        with torch.no_grad():
            out.append(_l2(model(**inputs).last_hidden_state[:, 0, :]).cpu().numpy())
    return np.vstack(out)


def vit_images(images, checkpoint: str = "google/vit-base-patch16-224",
               batch: int = 64) -> np.ndarray:
    """Supervised ViT class token, L2-normalized."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if "vit" not in _cache:
        _cache["vit"] = (AutoImageProcessor.from_pretrained(checkpoint, use_fast=False),
                         AutoModel.from_pretrained(checkpoint).to(_device()).eval())
    proc, model = _cache["vit"]
    out = []
    for i in range(0, len(images), batch):
        inputs = proc(images=[im.convert("RGB") for im in images[i:i + batch]],
                      return_tensors="pt").to(_device())
        with torch.no_grad():
            out.append(_l2(model(**inputs).last_hidden_state[:, 0, :]).cpu().numpy())
    return np.vstack(out)


# --------------------------------------------------------------------------
# Joint multimodal encoders. Each returns image and text towers separately so
# the caller can pair them; concatenation happens when pairs are assembled.
# --------------------------------------------------------------------------


def clip_towers(images, texts, checkpoint: str = "openai/clip-vit-base-patch32",
                batch: int = 32):
    """CLIP image and text towers through their projections, L2-normalized (512 + 512)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if "clip" not in _cache:
        _cache["clip"] = (CLIPProcessor.from_pretrained(checkpoint, use_fast=False),
                          CLIPModel.from_pretrained(checkpoint).to(_device()).eval())
    proc, model = _cache["clip"]
    img, txt = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            inputs = proc(images=list(images[i:i + batch]), return_tensors="pt").to(_device())
            pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            img.append(_l2(model.visual_projection(pooled)).cpu().numpy())
        for i in range(0, len(texts), batch):
            inputs = proc(text=list(texts[i:i + batch]), return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(_device())
            pooled = model.text_model(input_ids=inputs["input_ids"],
                                      attention_mask=inputs["attention_mask"]).pooler_output
            txt.append(_l2(model.text_projection(pooled)).cpu().numpy())
    return np.vstack(img), np.vstack(txt)


def flava_towers(images, texts, checkpoint: str = "facebook/flava-full", batch: int = 32):
    """FLAVA image and text class tokens, each L2-normalized (768 + 768)."""
    import torch
    from transformers import FlavaModel, FlavaProcessor

    if "flava" not in _cache:
        _cache["flava"] = (FlavaProcessor.from_pretrained(checkpoint, use_fast=False),
                           FlavaModel.from_pretrained(checkpoint).to(_device()).eval())
    proc, model = _cache["flava"]
    img, txt = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            inputs = proc(images=list(images[i:i + batch]), return_tensors="pt").to(_device())
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            img.append(_l2(feats.last_hidden_state[:, 0, :]).cpu().numpy())
        for i in range(0, len(texts), batch):
            inputs = proc(text=list(texts[i:i + batch]), return_tensors="pt",
                          padding=True, truncation=True, max_length=77).to(_device())
            feats = model.get_text_features(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"])
            txt.append(_l2(feats.last_hidden_state[:, 0, :]).cpu().numpy())
    return np.vstack(img), np.vstack(txt)


def siglip_joint(images, texts, checkpoint: str = "google/siglip-base-patch16-224",
                 batch: int = 32) -> np.ndarray:
    """SigLIP pooled tower outputs, L2-normalized, concatenated (768 + 768).

    Unlike the other two joint encoders this returns the concatenation directly,
    because SigLIP pairs are always assembled image-with-its-own-text.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    if "siglip" not in _cache:
        _cache["siglip"] = (AutoProcessor.from_pretrained(checkpoint, use_fast=False),
                            AutoModel.from_pretrained(checkpoint).to(_device()).eval())
    proc, model = _cache["siglip"]
    img, txt = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            inputs = proc(images=list(images[i:i + batch]), return_tensors="pt").to(_device())
            img.append(_l2(model.vision_model(
                pixel_values=inputs["pixel_values"]).pooler_output).cpu().numpy())
        for i in range(0, len(texts), batch):
            inputs = proc(text=list(texts[i:i + batch]), return_tensors="pt",
                          padding="max_length", truncation=True, max_length=64).to(_device())
            keep = {k: v for k, v in inputs.items()
                    if k in ("input_ids", "attention_mask", "position_ids")}
            txt.append(_l2(model.text_model(**keep).pooler_output).cpu().numpy())
    return np.hstack([np.vstack(img), np.vstack(txt)])


#: Dispatch from the encoder name in ``config.ENCODERS`` to its extraction call.
#: Text encoders take ``(texts)``; vision encoders take ``(images)``; joint
#: encoders take ``(images, texts)``.
EXTRACTORS = {
    "BERT":    lambda texts: masked_lm_first_token("bert-base-uncased", texts),
    "RoBERTa": lambda texts: masked_lm_first_token("roberta-base", texts),
    "mBERT":   lambda texts: masked_lm_first_token("bert-base-multilingual-cased", texts),
    "XLM-R":   lambda texts: masked_lm_first_token("xlm-roberta-base", texts),
    "MiniLM":  minilm_sentence,
    "CLIP":    clip_images,
    "DINOv2":  dinov2_images,
    "ViT":     vit_images,
    "CLIP-joint":   clip_towers,
    "FLAVA-joint":  flava_towers,
    "SigLIP-joint": siglip_joint,
}
