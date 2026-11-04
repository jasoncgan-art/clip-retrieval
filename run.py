"""End-to-end run: download -> embed -> retrieve -> evaluate -> report.

Pipeline (each step prints what it is doing, so a failed run says where):
  1. Load pairs (Flickr8k subset; CIFAR-10 demo fallback only with
     --demo, never silently — see data.py for why the distinction matters).
  2. Load pretrained CLIP ViT-B/32 (CPU by default) and embed images +
     captions, L2-normalized.
  3. Brute-force cosine retrieval + Recall@1/5/10 in BOTH directions,
     plus the analytical random baseline on the same N.
  4. Save embeddings to artifacts/ (so eval can be re-run without
     re-encoding) and metrics to artifacts/metrics.json.

Usage:
  python3 run.py [--n-pairs 300] [--data-dir data] [--demo]
                 [--model openai/clip-vit-base-patch32] [--batch-size 16]

Nothing here trains anything. The model is pretrained and frozen; this
script measures what zero-shot CLIP already does on the eval set.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import (DEFAULT_N_PAIRS, build_demo_pairs, load_flickr8k_subset,
                  load_saved_pairs, save_pairs)
from embed import MODEL_NAME, embed_images, embed_texts, load_model
from retrieve import evaluate, format_results

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--demo", action="store_true",
                        help="Use the labeled CIFAR-10 demo set instead "
                             "of Flickr8k (numbers are then DEMO numbers).")
    parser.add_argument("--reuse-data", action="store_true",
                        help="Skip download; use data/ already on disk.")
    args = parser.parse_args()

    # Step 1 — data. The source actually used is printed and saved, so
    # metrics can never be quoted without knowing which set they came from.
    # NOTE on consistency: fresh runs save pairs to data/ and then RELOAD
    # them from disk before embedding. The saved JPEGs are the eval input;
    # embedding straight from dataset-decoded pixels instead would measure
    # slightly different pixels (JPEG is lossy) and make fresh runs differ
    # from --reuse-data runs by a few borderline rankings. This was
    # observed 2026-10-25 (R@1 0.7233 fresh-decoded vs 0.7500 disk-loaded)
    # and fixed by always embedding the disk-loaded images.
    if args.reuse_data:
        images, captions, manifest = load_saved_pairs(args.data_dir)
        source = manifest[0]["source"] if manifest else "unknown"
        print(f"[data] Reusing {len(images)} pairs from {args.data_dir} "
              f"(source: {source})")
    elif args.demo:
        pairs, source = build_demo_pairs(n_pairs=args.n_pairs)
        print(f"[data] DEMO set: {len(pairs)} pairs from {source} "
              f"(NOT Flickr8k — label any numbers 'demo')")
        save_pairs(pairs, args.data_dir)
        # Reload from disk (see note above): the saved JPEGs are the
        # eval input, so fresh and --reuse-data runs measure identical
        # pixels instead of differing by a JPEG roundtrip.
        images, captions, _ = load_saved_pairs(args.data_dir)
    else:
        pairs, source = load_flickr8k_subset(n_pairs=args.n_pairs)
        print(f"[data] Flickr8k subset: {len(pairs)} pairs from {source}")
        save_pairs(pairs, args.data_dir)
        # Reload from disk (see note above): the saved JPEGs are the
        # eval input, so fresh and --reuse-data runs measure identical
        # pixels instead of differing by a JPEG roundtrip.
        images, captions, _ = load_saved_pairs(args.data_dir)

    # Step 2 — embed. Frozen pretrained model; no training happens here.
    import embed as embed_module
    embed_module.MODEL_NAME = args.model
    model, processor, device = load_model()
    print(f"[model] {args.model} on {device} (pretrained, frozen, zero-shot)")
    print(f"[embed] Encoding {len(images)} images ...")
    image_embs = embed_images(model, processor, images, device,
                              batch_size=args.batch_size)
    print(f"[embed] Encoding {len(captions)} captions ...")
    text_embs = embed_texts(model, processor, captions, device)

    # Step 3 — evaluate both directions + random baseline on the same N.
    results = evaluate(image_embs, text_embs)
    results["source"] = source
    results["model"] = args.model
    print("[eval] Recall@K (rows: direction; random = K/N baseline)")
    print(format_results(results))

    # Step 4 — artifacts: embeddings (reusable) + metrics (reportable).
    os.makedirs(args.artifacts_dir, exist_ok=True)
    np.save(os.path.join(args.artifacts_dir, "image_embs.npy"), image_embs)
    np.save(os.path.join(args.artifacts_dir, "text_embs.npy"), text_embs)
    with open(os.path.join(args.artifacts_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] Embeddings + metrics.json written to {args.artifacts_dir}/")


if __name__ == "__main__":
    main()
