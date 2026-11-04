"""Data loading for CLIP image-text retrieval.

Why this file exists: evaluation is only honest if the image <-> caption
pairing is exactly known. This module builds a small, explicit list of
(image, caption) pairs so that "the correct answer" for every query is
unambiguous: pair i's image matches pair i's caption, and no other.

Dataset (v1): a Flickr8k subset from Hugging Face, one caption per image.
Flickr8k ships 5 captions per image; we take only the FIRST caption so the
mapping is 1:1. With 5 captions per image, Recall@K would need a rule for
"any of the 5 counts as correct" — that is a valid choice too, but 1:1 is
simpler to explain in an interview and harder to get wrong.

Why no train/test split: v1 evaluates a PRETRAINED, zero-shot CLIP model.
Nothing is trained here, so nothing can leak from train into test. The
whole subset is one held-out eval set.

Fallback: if no Flickr8k source downloads, build_demo_pairs() constructs a
clearly-labeled demo set from CIFAR-10 with class-name captions
("a photo of a {class}"). Demo-set numbers must NEVER be quoted as
Flickr8k numbers — the loader records which source was actually used in
the pairs file it writes, and run.py prints it.
"""

import json
import os
import random

# Hugging Face dataset mirrors tried, in order. The first one that loads
# wins; the choice is recorded in the output metadata.
# NOTE (2026-10-25, verified by probing): the first choice in the original
# spec draft, "nlphuji/flickr8k", does not exist on the Hub. The working
# source is "jxie/flickr8k" (most-downloaded Flickr8k mirror): test split
# with keys image + caption_0..caption_4. Spec updated to match.
FLICKR8K_CANDIDATES = [
    "jxie/flickr8k",
]

DEFAULT_N_PAIRS = 300


def _first_caption(row):
    """Return the first caption for one dataset row, whatever its shape.

    Different Flickr8k mirrors store captions differently: sometimes as a
    list under 'captions'/'caption', sometimes as numbered fields
    caption_0..caption_4 (jxie/flickr8k), sometimes as a single string.
    This normalizes all of them to one string, so the rest of the code
    only ever sees one caption per image.
    """
    for key in ("caption_0", "captions", "caption", "text"):
        if key in row and row[key]:
            value = row[key]
            if isinstance(value, (list, tuple)):
                return str(value[0]).strip()
            return str(value).strip()
    raise ValueError(f"Row has no usable caption field. Keys: {list(row.keys())}")


def _image_from_row(row):
    """Return a PIL image for one dataset row, whatever its shape."""
    for key in ("image", "img"):
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError(f"Row has no usable image field. Keys: {list(row.keys())}")


def load_flickr8k_subset(n_pairs=DEFAULT_N_PAIRS, seed=0, cache_dir=None):
    """Load n_pairs (image, first-caption) pairs from Flickr8k via HF.

    Loads the test split NON-streaming (streaming mode crashed the
    interpreter at shutdown in this environment — a datasets/pyarrow
    thread-teardown quirk, not a data error — so it is avoided). Only
    the test split is downloaded, never the full 8,000-image set. Rows
    are shuffled with a fixed seed so the subset is stable and not just
    "the first 300 rows of one file".

    Returns (pairs, source_name) where pairs is a list of dicts with keys
    "image" (PIL image), "caption" (str), "source" (dataset name).
    Raises RuntimeError if no candidate source loads.
    """
    from datasets import load_dataset

    last_error = None
    for name in FLICKR8K_CANDIDATES:
        try:
            # Test split first (it is the standard held-out split); fall
            # back to train/validation only if a mirror lacks test.
            ds = None
            for split in ("test", "train", "validation"):
                try:
                    ds = load_dataset(name, split=split)
                    break
                except Exception:
                    continue
            if ds is None:
                raise RuntimeError(f"No usable split in {name}")
            rng = random.Random(seed)
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            pairs = []
            for i in idx[:n_pairs]:
                row = ds[i]
                pairs.append({
                    "image": _image_from_row(row),
                    "caption": _first_caption(row),
                    "source": name,
                })
            if len(pairs) < n_pairs:
                raise RuntimeError(f"{name}: only {len(pairs)} pairs available")
            return pairs, name
        except Exception as exc:  # try the next mirror
            last_error = exc
    raise RuntimeError(f"No Flickr8k source loaded. Last error: {last_error}")


def build_demo_pairs(n_pairs=DEFAULT_N_PAIRS, seed=0):
    """Fallback demo set: CIFAR-10 images with class-name captions.

    WHY this is honest: class names are the only text CIFAR-10 has, so a
    caption like "a photo of a dog" is a real (if easy) text query, and
    many images share one caption — which is exactly why demo numbers
    look inflated and must be labeled DEMO, never Flickr8k. The template
    "a photo of a ..." is CLIP's own canonical zero-shot prompt format.

    Returns (pairs, source_name) in the same shape as load_flickr8k_subset.
    """
    import numpy as np
    from datasets import load_dataset

    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    ds = load_dataset("cifar10", split="test")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pairs = []
    for i in idx[:n_pairs]:
        row = ds[i]
        label = classes[int(row["label"])]
        pairs.append({
            "image": row["img"],
            "caption": f"a photo of a {label}",
            "source": "cifar10-demo",
        })
    return pairs, "cifar10-demo"


def save_pairs(pairs, out_dir):
    """Save images + a pairs.json manifest (index, caption, source).

    The manifest is what evaluation reads back: it pins exactly which
    caption belongs to which image, so a later eval run cannot silently
    re-pair them. Images are saved as images/{i:04d}.jpg.
    """
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = []
    for i, pair in enumerate(pairs):
        fname = f"{i:04d}.jpg"
        pair["image"].convert("RGB").save(os.path.join(img_dir, fname))
        manifest.append({"index": i, "file": fname,
                         "caption": pair["caption"], "source": pair["source"]})
    with open(os.path.join(out_dir, "pairs.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_saved_pairs(data_dir):
    """Load (images, captions) back from a directory written by save_pairs.

    Returns (images, captions, manifest): images in manifest order, so
    image i's correct caption is captions[i]. This 1:1 order is the whole
    basis of the evaluation — never sort or shuffle one side alone.
    """
    from PIL import Image

    with open(os.path.join(data_dir, "pairs.json")) as f:
        manifest = json.load(f)
    images = [Image.open(os.path.join(data_dir, "images", m["file"])).convert("RGB")
              for m in manifest]
    captions = [m["caption"] for m in manifest]
    return images, captions, manifest
