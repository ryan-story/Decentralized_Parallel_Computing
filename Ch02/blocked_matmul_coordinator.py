"""Exact distributed matrix multiplication on a two-dimensional worker grid.

This is section 2.3.1. The output matrix `C` is tiled across a `w_r` by `w_c`
grid of workers, each worker owns one tile, and each worker computes its tile
from a row block of `A` and a column block of `B`. Both blocks carry the whole
contracted dimension `p`, so a worker finishes its tile alone: no partial values
are ever exchanged or summed.

The consequence is the one the chapter cares about. Because the tiles are
disjoint, what comes back across the network is exactly one copy of `C`, and it
stays exactly one copy however many workers there are. Splitting the contracted
dimension instead, which is what `distributed_gram_coordinator.py` is forced to
do, returns one COMPLETE `m` by `n` partial per worker, so the return traffic
grows with every worker added.

    bytes across the network, for one exact product

        2-D output tiling          w_c * |A| + w_r * |B| + |C|
        1-D contraction split      |A| + |B| + w * |C|

`MatrixMulKernel` is identical in both. Only the coordinator differs.

    python blocked_matmul_coordinator.py     # self-check against a centralized product
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import matmul_launcher

BYTES_PER_FLOAT = 4

# users staged onto the device at a time when building a Gram tile, chosen so an
# operand chunk stays well under a consumer GPU's memory
MAX_USERS_PER_LAUNCH = 2e7


def choose_worker_grid(w, m, n):
    """Factor `w` into `(w_r, w_c)`, favouring the grid with the smallest blocks.

    Each worker is sent `m / w_r` rows of `A` and `n / w_c` columns of `B`, both
    carrying the full contracted dimension, so what a worker must receive is
    proportional to `m / w_r + n / w_c`. Minimizing that sum keeps a worker from
    being handed one very tall block and one very thin one. On a square problem it
    picks the squarest grid; on a wide one it deliberately does not.
    """
    candidates = []
    for worker_rows in range(1, w + 1):
        if w % worker_rows == 0:
            worker_cols = w // worker_rows
            block_input = m / worker_rows + n / worker_cols
            candidates.append((block_input, worker_rows, worker_cols))
    _, worker_rows, worker_cols = min(candidates)
    return worker_rows, worker_cols


def blocked_matmul(A, B, device_ids=(0,), return_layout=False):
    """Exact `C = A B`, with the output tiled across a grid of `len(device_ids)` workers."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    m, p = A.shape
    if B.shape[0] != p:
        raise ValueError(f"inner dimensions disagree: A is {A.shape}, B is {B.shape}")
    n = B.shape[1]

    worker_rows, worker_cols = choose_worker_grid(len(device_ids), m, n)
    row_boundaries = np.linspace(0, m, worker_rows + 1, dtype=int)
    col_boundaries = np.linspace(0, n, worker_cols + 1, dtype=int)

    C = np.empty((m, n), dtype=np.float32)
    layout = []

    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        jobs, worker_id = [], 0
        for i in range(worker_rows):
            row_start, row_end = row_boundaries[i], row_boundaries[i + 1]
            for j in range(worker_cols):
                col_start, col_end = col_boundaries[j], col_boundaries[j + 1]

                A_i = A[row_start:row_end, :]      # full contracted dimension
                B_j = B[:, col_start:col_end]      # full contracted dimension

                future = executor.submit(matmul_launcher.matmul_launcher,
                                         device_ids[worker_id], A_i, B_j)
                jobs.append((future, row_start, row_end, col_start, col_end))
                layout.append({"worker": worker_id, "grid": (i, j),
                               "tile": (int(row_end - row_start), int(col_end - col_start))})
                worker_id += 1

        for future, row_start, row_end, col_start, col_end in jobs:
            C[row_start:row_end, col_start:col_end] = future.result()

    if return_layout:
        return C, {"grid": (worker_rows, worker_cols), "workers": layout}
    return C


def blocked_gram(R, device_ids=(0,), participants=None):
    """The item-item model `G = R^T R` computed by the blocked coordinator.

    A Gram matrix is just a product whose two operands are the same data, so this
    is `blocked_matmul(R.T, R)` with one change made for memory: the tile is formed
    from column slices of `R` directly rather than from a materialized `R.T`, which
    would be a second copy of the whole dataset.

    Worker `(I, J)` receives the columns of `R` for item block `I` and for item
    block `J`. Both slabs span every user, which is what lets the worker finish its
    tile alone. The tiles are disjoint, so the coordinator assembles them.

    `participants` drops workers to simulate failure. A dropped worker leaves its
    tile at zero, which is visibly different from the reduction case: a missing
    contribution does not shrink the answer, it punches a hole in it.
    """
    R = np.ascontiguousarray(R, dtype=np.float32)
    n_items = R.shape[1]
    w = len(device_ids)
    worker_rows, worker_cols = choose_worker_grid(w, n_items, n_items)
    rb = np.linspace(0, n_items, worker_rows + 1, dtype=int)
    cb = np.linspace(0, n_items, worker_cols + 1, dtype=int)
    live = set(range(w) if participants is None else participants)

    n_users = R.shape[0]
    # A worker's slabs span every user, so at catalogue scale they are far larger
    # than the tile they produce: 4,000 items over 162,539 users is a 1.3 GB operand
    # for an 8 MB result. The tile is therefore accumulated over chunks of users,
    # which is the same contraction the worker would do anyway, just staged so the
    # device never holds the whole slab at once.
    chunk = max(1, int(MAX_USERS_PER_LAUNCH // max(n_items, 1)))

    G = np.zeros((n_items, n_items), dtype=np.float32)
    worker_id = 0
    for I in range(worker_rows):
        for J in range(worker_cols):
            if worker_id in live:
                tile = np.zeros((rb[I + 1] - rb[I], cb[J + 1] - cb[J]), dtype=np.float32)
                for u0 in range(0, n_users, chunk):
                    u1 = min(u0 + chunk, n_users)
                    A_chunk = np.ascontiguousarray(R[u0:u1, rb[I]:rb[I + 1]].T)
                    B_chunk = np.ascontiguousarray(R[u0:u1, cb[J]:cb[J + 1]])
                    tile += matmul_launcher.matmul_launcher(
                        device_ids[worker_id], A_chunk, B_chunk)
                G[rb[I]:rb[I + 1], cb[J]:cb[J + 1]] = tile
            worker_id += 1
    return G


def gram_bytes(n_items, n_interactions, w):
    """Network bytes for the blocked Gram, in both directions.

    Outbound is the expensive half and it is easy to miss. Each item block of `R`
    is needed by every worker in its grid row and again by every worker in its grid
    column, so the coordinator ships `w_r + w_c` copies of the dataset. Inbound is
    the cheap half: the tiles are disjoint, so exactly one copy of `G` comes back
    however many workers there are.
    """
    worker_rows, worker_cols = choose_worker_grid(w, n_items, n_items)
    raw = int(n_interactions) * 2 * BYTES_PER_FLOAT
    out = (worker_rows + worker_cols) * raw
    back = int(n_items) * int(n_items) * BYTES_PER_FLOAT
    return {"grid": (worker_rows, worker_cols), "outbound": int(out),
            "inbound": int(back), "total": int(out + back)}


def recommend(G, liked_items, top_n=10, exclude_seen=True):
    """Score items for a user who liked `liked_items`, straight from the exact model.

    Item-item collaborative filtering, cosine-normalized by the model's own
    diagonal: `sim[i, j] = G[i, j] / sqrt(G[i, i] G[j, j])`, and a user's score for
    item `j` is the sum of `sim[i, j]` over the items `i` they already liked. The
    normalizer is the item-count diagonal that section 2.4.1 says has to be exact;
    here it is, because this model is exact.
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


def inbound_bytes(m, n, p, w):
    """What crosses the network for one exact product under the output tiling.

    The `results_back` term is the whole point: it is `m * n * 4` regardless of `w`,
    because the tiles are disjoint and together they are exactly `C`.
    """
    worker_rows, worker_cols = choose_worker_grid(w, m, n)
    F = BYTES_PER_FLOAT
    operands_out = worker_cols * m * p * F + worker_rows * p * n * F
    results_back = m * n * F
    return {"grid": (worker_rows, worker_cols),
            "operands_out": int(operands_out),
            "results_back": int(results_back),
            "total": int(operands_out + results_back)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"CUDA kernel path live: {matmul_launcher.gpu_available()}\n")
    for (m, n, p), w in [((256, 192, 320), 4), ((300, 300, 128), 6), ((129, 257, 64), 8)]:
        A = rng.standard_normal((m, p)).astype(np.float32)
        B = rng.standard_normal((p, n)).astype(np.float32)
        C, lay = blocked_matmul(A, B, [0] * w, return_layout=True)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = np.abs(C - ref).max() / np.abs(ref).max()
        b = inbound_bytes(m, n, p, w)
        tiles = sum(t["tile"][0] * t["tile"][1] for t in lay["workers"])
        print(f"  m={m:4} n={n:4} p={p:4} w={w:2}  grid={lay['grid']}  rel err={err:.2e}  "
              f"tiles cover C: {tiles == m * n}  back={b['results_back']/1e6:.2f} MB")
