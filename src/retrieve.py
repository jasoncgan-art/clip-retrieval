"""Retrieval + evaluation, both directions.

Why brute force here: with N in the hundreds, the full N x N similarity
matrix is tiny (300 x 300 = 90,000 dot products) and EXACT — it returns
the true nearest neighbours, no approximation error. Approximate nearest
neighbour search (FAISS/ANN) only becomes necessary at catalog scale
(millions of items), where brute force is too slow. Claiming ANN here
would be theatre; the honest production line is: "brute force at this
scale, ANN is the next step at scale" — and ANN is a v2 item.

Why both directions: image-to-text and text-to-image are different tasks
with different difficulty. A model can be good at one and weak at the
other (e.g. if captions are vague, text-to-image suffers). Reporting
only the better direction would be cherry-picking, so both are always
reported together.

Why Recall@K: retrieval has no single "accuracy" — the question is
whether the ONE correct match lands in the top K results. Recall@K is
exactly that fraction, averaged over all queries. K = 1 is strict
(the correct match must be ranked first); K = 5 and 10 show how often
it is at least near the top.

Baseline: random retrieval ranks the correct match uniformly at random,
so expected Recall@K = K / N. With N = 300, random Recall@5 = 1.67%.
Any real number must be compared against that same-N figure.
"""

import numpy as np

KS = (1, 5, 10)


def similarity_matrix(image_embs, text_embs):
    """Return the (N_images, N_texts) cosine-similarity matrix.

    Because both embedding sets are L2-normalized (see embed.py), the dot
    product IS cosine similarity — no extra division needed. Row i,
    column j = similarity between image i and caption j. The diagonal
    (i == j) holds the correct pairs; everything off-diagonal is a
    distractor. A good model puts the diagonal on top of each row/column.
    """
    return image_embs @ text_embs.T


def recall_at_k(sim_matrix, ks=KS):
    """Recall@K in the direction "rows are queries, columns are ranked".

    For each query row, rank all columns by similarity (highest first)
    and check whether the correct column — the one with the same index
    as the query — is within the top K. Returns {K: recall}.

    Called twice by evaluate(): once with sim_matrix (image queries over
    caption columns = image-to-text) and once with sim_matrix.T (text
    queries over image columns = text-to-image). Same function, both
    directions — no duplicated logic to drift apart.
    """
    n = sim_matrix.shape[0]
    # argsort ascending, then flip: rank 0 = highest similarity.
    ranking = np.argsort(sim_matrix, axis=1)[:, ::-1]
    correct = np.arange(n)[:, None]  # correct column for each query row
    recalls = {}
    for k in ks:
        # Is the correct column anywhere in the row's top-K columns?
        hits = (ranking[:, :k] == correct).any(axis=1)
        recalls[k] = float(hits.mean())
    return recalls


def random_baseline(n, ks=KS):
    """Expected Recall@K of random ranking: K / N for each K.

    Analytical, not simulated: a random ranking puts the correct item
    uniformly anywhere in N positions, so P(in top K) = K / N exactly.
    """
    return {k: k / n for k in ks}


def evaluate(image_embs, text_embs, ks=KS):
    """Full evaluation. Returns a dict with both directions + baseline.

    Keys: "image_to_text" {K: recall}, "text_to_image" {K: recall},
    "random_baseline" {K: recall}, "n_pairs" int.
    """
    sim = similarity_matrix(image_embs, text_embs)
    n = sim.shape[0]
    return {
        "image_to_text": recall_at_k(sim, ks),
        "text_to_image": recall_at_k(sim.T, ks),
        "random_baseline": random_baseline(n, ks),
        "n_pairs": n,
    }


def format_results(results):
    """One human-readable block for stdout and for the README table."""
    lines = [f"Eval pairs (N): {results['n_pairs']}"]
    lines.append(f"{'Direction':<16}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    for name in ("image_to_text", "text_to_image", "random_baseline"):
        r = results[name]
        lines.append(f"{name:<16}{r[1]:>8.4f}{r[5]:>8.4f}{r[10]:>8.4f}")
    return "\n".join(lines)
