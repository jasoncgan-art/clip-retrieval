# CLIP Vision-Language Retrieval

Zero-shot image↔text retrieval with pretrained CLIP (ViT-B/32), evaluated
honestly in **both directions** against a random baseline — same evaluation
discipline as the other portfolio repos.

**Built ≠ mastered.** This repo was built locally as a study artifact. It
must be studied line-by-line under guidance and defended in a grill
before it is ever listed on a resume. The model here is **pretrained by
OpenAI, not trained by me** — the skill on display is using a dual
encoder correctly and evaluating it honestly, not "I trained CLIP".

## Architecture

```
caption ──► text encoder (Transformer) ──┐
                                          ├─► shared 512-dim space ──► cosine similarity ──► top-K
image ────► image encoder (ViT-B/32) ────┘        (L2-normalized vectors)
```

**Why CLIP works for retrieval — the dual-encoder idea.** CLIP trains two
encoders together with a contrastive loss: in each training batch of N
image–text pairs, the N correct pairs are pulled together in a shared
embedding space while the N²−N incorrect pairings are pushed apart.
After pretraining, an image and its matching caption land close together
even though they came from different encoders — so retrieval is just
nearest neighbours across modalities. This project *uses* that space; it
does not retrain the loss.

**Why L2-normalize:** cosine similarity equals the dot product only for
unit-length vectors. Normalizing first means retrieval is one matrix
multiplication and no long vector can dominate by magnitude alone.

**Why brute force here:** at N = 300 the full similarity matrix is
90,000 dot products — exact, no approximation error. Approximate nearest
neighbour search (FAISS/ANN) only becomes necessary at catalog scale
(millions of items). ANN is the stated production-scale next step, not
something claimed here.

## Measured results (actually run, CPU, 2026-10-25)

- **Model:** `openai/clip-vit-base-patch32` via `transformers` 5.20,
  pretrained and frozen (zero-shot — nothing trained).
- **Data:** Flickr8k subset, **N = 300 pairs** from the `jxie/flickr8k`
  test split (seed 0, first caption `caption_0` per image, 1:1 pairing).
  Say "Flickr8k subset, 300 pairs" — never bare "Flickr8k", which would
  imply the full 8,000-image benchmark.
- **Baseline:** random retrieval, expected Recall@K = K/N on the same N.

| Direction | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| Image → text | **0.7500** | **0.9600** | **0.9867** |
| Text → image | **0.7233** | **0.9367** | **0.9767** |
| Random baseline (K/300) | 0.0033 | 0.0167 | 0.0333 |

Sanity checks that passed: embeddings are (300, 512) and unit-norm
(min/max norm 1.0000); mean diagonal similarity 0.3112 vs off-diagonal
0.1649 — correct pairs really do score higher. Both directions are
reported together, always: reporting only the better one would be
cherry-picking.

**Reproducibility note (found during verification):** the first fresh
run scored R@1 0.7233 while disk-loaded runs scored 0.7500, because the
fresh path embedded dataset-decoded pixels while the reuse path embedded
JPEG-roundtripped pixels (JPEG is lossy). Fixed by always embedding the
disk-saved images — `run_all.sh` and `--reuse-data` now produce
bit-identical `metrics.json`. The table above is from the fixed pipeline,
verified by a clean `run_all.sh` run followed by a `--reuse-data` run
with identical output.

## Why this small scale is honest

- A few hundred pairs is a CPU choice, not a production claim. The
  pipeline (encode → normalize → cosine → Recall@K both ways) is exactly
  the pipeline that scales; only the index (brute force → ANN) and the
  eval size change at scale.
- Zero-shot CLIP is the baseline, not the achievement. Fine-tuning is
  v2 work (see below), and no fine-tuned number is claimed here.
- One caption per image keeps the mapping 1:1, so Recall@K is
  unambiguous. Using all 5 Flickr captions per image would need an
  "any-of-5 counts" rule — valid, but easier to get wrong.

## How to run

```bash
./run_all.sh            # venv + install + full pipeline (300 pairs)
python3 run.py --n-pairs 300                  # same, with venv active
python3 run.py --reuse-data                  # re-eval saved data/ only
python3 run.py --demo                        # labeled CIFAR-10 demo set
                                             # (demo numbers ≠ Flickr numbers)
```

Outputs: `data/` (images + `pairs.json` manifest), `artifacts/`
(`image_embs.npy`, `text_embs.npy`, `metrics.json`). Both are gitignored;
only code, docs, and scripts are committed.

## Files

| File | What it does (and why it exists) |
|---|---|
| `src/data.py` | Flickr8k subset loading (1 caption/image) + CIFAR-10 demo fallback; save/load of the pairs manifest that pins the pairing |
| `src/embed.py` | CLIP image/text encoders → L2-normalized 512-dim embeddings (handles transformers 5.x returning `pooler_output` instead of a tensor) |
| `src/retrieve.py` | Brute-force cosine matrix, Recall@K both directions, analytical random baseline |
| `run.py` | End-to-end: data → embed → retrieve → evaluate → artifacts |
| `run_all.sh` | Clean reproduction: venv, CPU torch install, requirements, `run.py` |
| `requirements.txt` | torch (CPU-first via `run_all.sh`), transformers, datasets, Pillow, numpy, tqdm |

No notebook: none was needed, and a placeholder notebook would be worse
than none. `notebooks/` is kept empty for a future failure-analysis
notebook (v2).

## What changed from the previous README (kept intent, corrected plan)

The previous README planned Flickr30K + LoRA fine-tuning on Colab GPU.
That plan is **deferred to v2, not dropped**: this v1 delivers its
milestone 1 (zero-shot retrieval baseline, Recall@1/5/10) on CPU with a
small Flickr8k subset, because the build brief required a small
CPU-feasible build verified locally. Its standing rules are kept:
training (when it happens) belongs on a GPU, and the resume is updated
only after the defense drill is passed. Previous requirements listed
`open_clip_torch` and `scikit-learn`; v1 uses `transformers` instead
(the brief specified that route) and drops scikit-learn (Recall@K is a
few lines of numpy). Pillow / numpy / tqdm are kept.

## v2 (not built, not claimed)

- LoRA fine-tuning on Flickr30K (GPU/Colab) vs this zero-shot baseline.
- FAISS/ANN retrieval at larger scale, with ANN-vs-brute-force agreement.
- Failure-analysis gallery: cases where top-1 is wrong, and why.
- 5-caption evaluation ("any-of-5 counts") as an ablation.
