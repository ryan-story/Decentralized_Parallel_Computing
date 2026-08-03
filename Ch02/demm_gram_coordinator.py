"""Chapter 2, section 2.4.2: the DeMM Gram coordinator.

The same coordinator as `distributed_gram_coordinator`, with one change: workers
return a compact summary instead of a full partial product.

    exact baseline   worker r returns  R_r^T R_r        [n_items, n_items]
    DeMM             worker r returns  (L_r, diag_r)    [n_items, q] + [n_items]

The merge is still an associative sum. The diagonals add exactly, and the factors
combine by concatenation, because summing outer products stacks their factors:

    sum_r L_r L_r^T  =  [L_0 | L_1 | ...] [L_0 | L_1 | ...]^T

so the merged model is just the collection of factors. Nothing here reconstructs
the [n_items, n_items] matrix. Section 2.4.2 shows `C_hat += L_r @ M_r_T` because
it states plainly what the factors mean, but a deployment does not pay that: a
query goes through the factors as `sum_r L_r (L_r^T y)`, which is what `recommend`
below does, so the full matrix is never formed on any machine.

Deviations from the book's listing, and why:

  * `participants` lets the app drop a worker and watch its term leave the sum.
  * `demm_gram` returns the factors rather than assembling `C_hat`. The book shows
    the assembly because it states plainly what the factors mean, and then says a
    deployment should not pay for it; this is that deployment. `merged_gram` will
    assemble it if you want to measure reconstruction error, and `recommend` goes
    straight through the factors as the book describes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from demm_launcher import demm_launcher
from distributed_gram_coordinator import shard_boundaries

BYTES_PER_FLOAT = 4


def demm_gram(R, device_ids=(0,), q=32, s=8, h=1, seed=1000, participants=None):
    """Build the item-item model as a merged DeMM summary.

    `participants` optionally restricts which workers report, which is how the app
    shows what a missing worker costs. A worker that does not report simply drops
    its term from the sum.
    """
    R = np.asarray(R, dtype=np.float32)
    n_users, n_items = R.shape

    # Exactly the blocked-matmul setup of section 2.3.2, so the DeMM coordinator
    # and the exact one hand their workers the same blocks.
    A = R.T                                    # [n_items, n_users]
    B = R                                      # [n_users, n_items]

    boundaries = shard_boundaries(n_users, len(device_ids))
    chosen = range(len(device_ids)) if participants is None else sorted(participants)

    with ThreadPoolExecutor(max_workers=max(len(device_ids), 1)) as executor:
        futures = []
        for worker_id in chosen:
            s_r = boundaries[worker_id]
            e_r = boundaries[worker_id + 1]

            A_r = np.ascontiguousarray(A[:, s_r:e_r])     # this worker's user columns
            B_r = np.ascontiguousarray(B[s_r:e_r, :])     # the matching user rows

            futures.append(executor.submit(
                demm_launcher, device_ids[worker_id % len(device_ids)],
                A_r, B_r, q, s, h, seed + worker_id))     # a different seed each

        summaries = [f.result() for f in futures]

    factors = [L_r for L_r, _ in summaries]
    diag = np.zeros(n_items, dtype=np.float32)
    for _, d_r in summaries:
        diag = diag + d_r                                  # exact diagonals sum

    return {
        "factors": factors,
        "diag": diag,
        "n_items": int(n_items),
        "q": int(q),
        "workers_reporting": list(chosen),
        "n_workers": len(device_ids),
    }


def inbound_bytes(n_items, q, n_workers):
    """Bytes the coordinator receives: a [n_items, q] factor plus an [n_items]
    diagonal from each worker. Fixed in the number of users, like the exact
    exchange, but set by the RANK rather than by the catalogue squared."""
    per_worker = int(n_items) * int(q) * BYTES_PER_FLOAT + int(n_items) * BYTES_PER_FLOAT
    return per_worker * int(n_workers)


def merged_gram(model):
    """Materialize the approximate model. Only for measuring reconstruction error:
    scoring never calls this, which is the whole point of keeping the factors."""
    n = model["n_items"]
    G_hat = np.zeros((n, n), dtype=np.float32)
    for L_r in model["factors"]:
        G_hat += L_r @ L_r.T
    np.fill_diagonal(G_hat, model["diag"])                 # the exact normalizer
    return G_hat


def recommend(model, liked_items, top_n=10, exclude_seen=True):
    """Score items for a user who liked `liked_items`, straight from the factors.

    With `y` the user's interaction vector and `d` the exact diagonal, the cosine
    normalization divides by `sqrt(d)` on the way in and again on the way out,
    and the model is applied as `sum_r L_r (L_r^T y)`. The [n_items, n_items]
    matrix is never built.
    """
    diag = model["diag"]
    inv = (1.0 / np.sqrt(np.clip(diag, 1e-9, None))).astype(np.float32)

    liked = np.asarray(list(liked_items), dtype=int)
    if liked.size == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    y = np.zeros(model["n_items"], dtype=np.float32)
    y[liked] = 1.0
    y = y * inv                                            # scale in

    scores = np.zeros(model["n_items"], dtype=np.float32)
    for L_r in model["factors"]:
        scores += L_r @ (L_r.T @ y)                        # G y through the factors
    scores = scores * inv                                  # scale out

    if exclude_seen:
        scores[liked] = -np.inf

    top = np.argpartition(-scores, min(top_n, scores.size - 1))[:top_n]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]
