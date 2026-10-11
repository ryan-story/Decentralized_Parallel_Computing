"""Chapter 2, section 2.4: DeMM applied to the item-item model.

Section 2.4.1 sets A = R^T and B = R and cuts the contracted dimension, the users:
worker r receives its own users' rows R_r and never sees anyone else's. Its partial
product R_r^T R_r is symmetric, so the left and right factors describe the same
subspace and a worker sends one factor, L_r, instead of two.

    demm_gram_worker(device_id, R_r, q, s, h, _SRC, seed)
        one worker: Algorithm 2.1 (listing 2.8) with A_r = R_r^T and B_r = R_r,
        returning L_r and the exact item counts diag(R_r^T R_r)
    demm_gram(R, device_ids, q, ...)
        the coordinator: one user shard per worker, factors kept, counts summed
    inbound_bytes(n_items, q, n_workers)
        what the workers send back
    recommend(model, liked)
        recommendations straight from the factors, without forming the model

The exact item counts. The recommender divides every co-occurrence by
sqrt(count_i * count_j), so those counts are its normalizer. A worker sends them
exactly, one number per item (16 KB for 4,000 items), rather than letting them be
approximated along with everything else: a rank-q fit reproduces the large
diagonal entries and distorts the normalized scores, and at MovieLens 25M scale
that costs most of the quality Table 2.1 reports. The counts are included in the
returned bytes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from demm_coordinator import _SRC
from demm_launcher import demm_launcher

BYTES_PER_FLOAT = 4


def shard_boundaries(p, n_workers):
    """Contiguous shard boundaries over the shared dimension, as listing 2.9 cuts it."""
    return np.linspace(0, p, n_workers + 1, dtype=int)


def demm_gram_worker(device_id, R_r, q, s, h, _SRC, seed):
    """One worker's summary of R_r^T R_r: the factor L_r and the exact item counts."""
    L_r, _ = demm_launcher(device_id, R_r.T, R_r, q, s, h, _SRC, seed)   # M_r = L_r
    diag_r = (R_r * R_r).sum(axis=0).astype(np.float32)
    return L_r.astype(np.float32), diag_r


def demm_gram(R, device_ids=(0,), q=32, s=8, h=1, seed=1000, participants=None):
    """The item-item model as a collection of per-worker factors.

    `participants` restricts which workers report: a missing worker removes only
    its own users' term from the sum.
    """
    R = np.ascontiguousarray(R, dtype=np.float32)
    n_users, n_items = R.shape
    boundaries = shard_boundaries(n_users, len(device_ids))
    chosen = range(len(device_ids)) if participants is None else sorted(participants)

    with ThreadPoolExecutor(max_workers=max(len(device_ids), 1)) as executor:
        futures = []
        for worker_id in chosen:
            R_r = R[boundaries[worker_id]:boundaries[worker_id + 1]]
            futures.append(executor.submit(
                demm_gram_worker, device_ids[worker_id], R_r, q, s, h, _SRC,
                seed + worker_id))
        summaries = [f.result() for f in futures]

    diag = np.zeros(n_items, dtype=np.float32)
    for _, d_r in summaries:
        diag += d_r

    return {"factors": [L_r for L_r, _ in summaries], "diag": diag,
            "n_items": int(n_items), "q": int(q),
            "workers_reporting": list(chosen), "n_workers": len(device_ids)}


def inbound_bytes(n_items, q, n_workers):
    """Bytes back: an n_items by q factor plus n_items exact counts, per worker."""
    per_worker = int(n_items) * int(q) * BYTES_PER_FLOAT + int(n_items) * BYTES_PER_FLOAT
    return per_worker * int(n_workers)


def merged_gram(model):
    """The approximate model, assembled. Only for measurement; scoring never needs it."""
    n = model["n_items"]
    G_hat = np.zeros((n, n), dtype=np.float32)
    for L_r in model["factors"]:
        G_hat += L_r @ L_r.T
    np.fill_diagonal(G_hat, model["diag"])
    return G_hat


def recommend(model, liked_items, top_n=10, exclude_seen=True):
    """Recommendations straight from the factors: G y = sum_r L_r (L_r^T y).

    Same cosine normalization as the exact model, using the exact item counts.
    The n_items by n_items model is never formed.
    """
    inv = (1.0 / np.sqrt(np.clip(model["diag"], 1e-9, None))).astype(np.float32)
    liked = np.asarray(list(liked_items), dtype=int)
    if liked.size == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    y = np.zeros(model["n_items"], dtype=np.float32)
    y[liked] = 1.0
    y *= inv
    scores = np.zeros(model["n_items"], dtype=np.float32)
    for L_r in model["factors"]:
        scores += L_r @ (L_r.T @ y)
    scores *= inv
    if exclude_seen:
        scores[liked] = -np.inf

    top = np.argpartition(-scores, min(top_n, scores.size - 1))[:top_n]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]
