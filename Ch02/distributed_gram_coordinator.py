"""Chapter 2, section 2.3.2: the distributed Gram coordinator.

This is the chapter's exact baseline. The item-item model is a Gram matrix,

    G = R^T R,

whose contraction runs over users, so sharding the users across workers turns it
into a blocked matmul with `A = R^T` and `B = R`:

    G = sum_r  R_r^T R_r.

The coordinator splits the contracted dimension, hands each worker its blocks,
and adds the full-size partial products it gets back. The result is exact.

It is also expensive, and that is the point of having it. Each worker returns a
complete `n_items` by `n_items` matrix, so the bytes the coordinator receives are
set by the size of the catalogue rather than by how much data anybody holds:
adding users increases local work and changes the exchange not at all. At 4,000
items and four-byte floats that is 64 MB per worker, 512 MB across eight.
`inbound_bytes` below is that accounting, and section 2.4 is the answer to it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from matmul_launcher import matmul_launcher

BYTES_PER_FLOAT = 4


def shard_boundaries(p, n_workers):
    """Split the contracted dimension into `n_workers` disjoint ranges.

    Returns the boundary indices, so worker `r` owns `[boundaries[r], boundaries[r+1])`.
    This is `s_r` and `e_r` from equation 2.7.
    """
    return np.linspace(0, p, n_workers + 1, dtype=int)


def distributed_gram(R, device_ids=(0,), return_partials=False, participants=None):
    """Build the exact item-item model by blocked matmul over the user dimension.

    `R` is the [n_users, n_items] interaction matrix. Users are the contracted
    dimension, so each worker receives a block of user ROWS and contributes the
    full-size partial `R_r^T R_r`.

    `participants` optionally restricts which workers report. Because the merge is
    a sum, a worker that does not report simply drops its term, so the model still
    reflects everyone who did. That is a property of the additive merge and not of
    any compression, which is why the baseline has it too.
    """
    R = np.asarray(R, dtype=np.float32)
    A = R.T                                    # [n_items, n_users]
    B = R                                      # [n_users, n_items]

    p = A.shape[1]                             # the contracted dimension: users
    boundaries = shard_boundaries(p, len(device_ids))
    chosen = range(len(device_ids)) if participants is None else sorted(participants)

    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = []
        for worker_id in chosen:
            device_id = device_ids[worker_id % len(device_ids)]
            s_r = boundaries[worker_id]
            e_r = boundaries[worker_id + 1]

            A_r = np.ascontiguousarray(A[:, s_r:e_r])     # this worker's user columns
            B_r = np.ascontiguousarray(B[s_r:e_r, :])     # the matching user rows

            futures.append(executor.submit(matmul_launcher, device_id, A_r, B_r))

        partials = [f.result() for f in futures]

    G = sum(partials)                          # the merge: an associative sum
    return (G, partials) if return_partials else G


def inbound_bytes(n_items, n_workers):
    """Bytes the coordinator receives: one full [n_items, n_items] partial per worker.

    Independent of the number of users, which is exactly what makes it expensive:
    the cost is the catalogue, not the data.
    """
    return int(n_items) * int(n_items) * BYTES_PER_FLOAT * int(n_workers)


def raw_interaction_bytes(n_interactions):
    """Bytes to give up on decentralizing and ship the raw interactions instead,
    one (item, value) pair each. Grows with the users, unlike the Gram exchange."""
    return int(n_interactions) * 2 * BYTES_PER_FLOAT


def recommend(G, liked_items, top_n=10, exclude_seen=True):
    """Score items for a user who liked `liked_items`, straight from the exact model.

    Item-item collaborative filtering, cosine-normalized by the model's own
    diagonal: `sim[i, j] = G[i, j] / sqrt(G[i, i] G[j, j])`, and a user's score for
    item `j` is the sum of `sim[i, j]` over the items `i` they already liked.
    """
    G = np.asarray(G, dtype=np.float32)
    inv = (1.0 / np.sqrt(np.clip(np.diag(G), 1e-9, None))).astype(np.float32)
    liked = np.asarray(list(liked_items), dtype=int)
    if liked.size == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    scores = ((G[liked] * inv[liked][:, None]).sum(axis=0) * inv).astype(np.float32)
    if exclude_seen:
        scores[liked] = -np.inf

    top = np.argpartition(-scores, min(top_n, scores.size - 1))[:top_n]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]
