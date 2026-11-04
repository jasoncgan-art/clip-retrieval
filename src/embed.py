"""CLIP embedding: image and text encoders into one shared space.

Why CLIP works for retrieval: CLIP is a DUAL ENCODER. An image encoder
(a Vision Transformer, ViT) and a text encoder (a Transformer) are trained
together with a contrastive loss: in each training batch, the N correct
image-text pairs are pulled together in a shared embedding space while
the N^2 - N incorrect pairings are pushed apart. After pretraining, an
image and its matching caption land close together in that space even
though they came from different encoders — so retrieval is just
"nearest neighbours across modalities".

Why L2-normalize: cosine similarity between two vectors equals their dot
product ONLY when both are unit length. Normalizing here means the
retrieval step can be one matrix multiplication, and no long vector
(e.g. a bright, high-contrast image) can dominate purely by magnitude.

Why this file uses transformers: `openai/clip-vit-base-patch32` is the
smallest standard CLIP release (ViT-B/32: image split into 32x32 patches,
512-dim joint space). It runs on CPU for a few hundred images, and it is
the model interviewers will recognize. We load it PRETRAINED and never
update its weights in v1 — this project evaluates zero-shot CLIP; it
does not train CLIP. Fine-tuning is a v2 item (see README).
"""

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM = 512  # joint image/text space of ViT-B/32


def _to_embedding(out, name):
    """Extract the (batch, 512) embedding tensor from an encoder call.

    transformers 4.x returned a plain tensor from get_*_features; 5.x
    returns a BaseModelOutputWithPooling whose pooler_output IS the
    projected 512-dim embedding (verified: shape (batch, 512), while the
    raw pooled vision output would be 768-dim). Accept both so the code
    keeps working across versions instead of breaking on an upgrade.
    """
    if torch.is_tensor(out):
        return out
    if hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
        return out.pooler_output
    raise TypeError(f"{name} returned unexpected type {type(out).__name__}")


def load_model(device=None):
    """Load the pretrained CLIP model + processor. Returns (model, processor, device).

    device defaults to CPU: this project is deliberately CPU-feasible.
    torch.no_grad is used at encode time (see below) because no gradients
    are ever needed — inference only, which also halves memory use.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # disables dropout etc.; inference must be deterministic
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor, device


@torch.no_grad()
def embed_images(model, processor, images, device, batch_size=16):
    """Encode images -> (N, 512) L2-normalized numpy array.

    processor handles CLIP's exact preprocessing (resize, center-crop,
    normalize with CLIP's mean/std) — never hand-roll this, because the
    model was trained on exactly this preprocessing and anything else
    silently degrades embeddings. Batching bounds memory on a small CPU
    box; batch_size=16 is conservative for ViT-B/32.
    """
    feats = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        # Images only: the processor returns just pixel_values here.
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = _to_embedding(
            model.get_image_features(pixel_values=pixel_values), "image")
        vec = out / out.norm(p=2, dim=-1, keepdim=True)  # L2-normalize
        feats.append(vec.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def embed_texts(model, processor, texts, device, batch_size=32):
    """Encode captions -> (N, 512) L2-normalized numpy array.

    Same idea as embed_images, other encoder: the text Transformer reads
    each caption (tokenized + padded by the processor) and outputs one
    512-dim vector per caption, L2-normalized so dot product = cosine.
    Text encoding is cheap, so a larger batch is fine.
    """
    feats = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        # Text only: the processor returns input_ids + attention_mask.
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        out = _to_embedding(
            model.get_text_features(input_ids=input_ids,
                                    attention_mask=attention_mask), "text")
        vec = out / out.norm(p=2, dim=-1, keepdim=True)
        feats.append(vec.cpu().numpy())
    return np.concatenate(feats, axis=0)
