"""Chapter 2, section 2.2.2 and section 2.4: the blocked coordinator.

    blocked_matmul(A, B, device_ids, b_m, b_n)   listing 2.7. Cut the output into a
                                                 b_m by b_n grid of blocks C(I, J),
                                                 give each worker the slabs A(I) and
                                                 B(J), and write each returned block
                                                 into place.

and the same coordinator applied to the project of section 2.4:

    blocked_gram(R, device_ids)   the item-item model G = R^T R with A = R^T and
                                  B = R (section 2.4.1). Exact.
    gram_bytes(...)               what crosses the network, both directions.
    recommend(G, liked)           item-item recommendations from the exact model.

The book writes listing 2.7 as a program with `device_ids = [0, 1, 2, 3]` and
`b_m = b_n = 2` at module level. Here the same statements are the body of
`blocked_matmul`, with those values as defaults, so the notebook can import it.
Device IDs may repeat: on a one-GPU machine `[0] * 4` gives four workers that
take turns on the same device.

    python blocked_matmul_coordinator.py     # checks against a centralized product
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from matmul_launcher import gpu_available, matmul_launcher

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "matmulkernel.cu"), encoding="utf-8") as _f:
    _SRC = _f.read()

BYTES_PER_FLOAT = 4


def blocked_matmul(A, B, device_ids=(0, 1, 2, 3), b_m=2, b_n=2, _SRC=_SRC):
    m = A.shape[0]
    p = A.shape[1]
    n = B.shape[1]

    row_boundaries = np.linspace(
        0,
        m,
        b_m + 1,
        dtype=int,
    )

    col_boundaries = np.linspace(
        0,
        n,
        b_n + 1,
        dtype=int,
    )

    with ThreadPoolExecutor(
        max_workers=len(device_ids)
    ) as executor:

        futures = []

        worker_id = 0

        for I in range(b_m):
            s_I = row_boundaries[I]
            e_I = row_boundaries[I + 1]
            A_I = A[s_I:e_I, :]

            for J in range(b_n):
                s_J = col_boundaries[J]
                e_J = col_boundaries[J + 1]
                B_J = B[:, s_J:e_J]

                future = executor.submit(
                    matmul_launcher,
                    device_ids[worker_id],
                    A_I,
                    B_J,
                    _SRC
                )

                futures.append((future, s_I, e_I, s_J, e_J))

                worker_id += 1

    C = np.empty((m, n), dtype=np.float32)

    for (future, s_I, e_I, s_J, e_J) in futures:
        C[s_I:e_I, s_J:e_J] = future.result()

    return C


# ------------------------------------------------------------- section 2.4

def choose_worker_grid(w, m, n):
    """Factor `w` workers into a `(b_m, b_n)` grid, favouring the smallest slabs.

    Each worker receives `m / b_m` rows of A and `n / b_n` columns of B, both
    spanning the whole shared dimension, so its input is proportional to
    `m / b_m + n / b_n`. Eight workers over the square item-item model become
    a 2 by 4 grid.
    """
    candidates = []
    for rows in range(1, w + 1):
        if w % rows == 0:
            cols = w // rows
            candidates.append((m / rows + n / cols, rows, cols))
    _, rows, cols = min(candidates)
    return rows, cols


# users staged onto the device at a time when forming one Gram block, so a slab
# that spans 162,539 users never has to sit in GPU memory whole
MAX_USERS_PER_LAUNCH = 2e7


def blocked_gram(R, device_ids=(0,), participants=None):
    """The item-item model G = R^T R from the blocked coordinator.

    Block (I, J) needs A(I) = R[:, item block I]^T and B(J) = R[:, item block J].
    Both span every user, so a worker finishes its block alone and the blocks are
    disjoint. At MovieLens 25M scale a slab is far larger than the block it
    produces (4,000 items over 162,539 users), so each block is accumulated over
    chunks of users. That is the same sum over users, staged so the device never
    holds a whole slab at once.

    `participants` drops workers: a missing worker leaves its block at zero.
    """
    R = np.ascontiguousarray(R, dtype=np.float32)
    n_users, n_items = R.shape
    b_m, b_n = choose_worker_grid(len(device_ids), n_items, n_items)
    rb = np.linspace(0, n_items, b_m + 1, dtype=int)
    cb = np.linspace(0, n_items, b_n + 1, dtype=int)
    live = set(range(len(device_ids)) if participants is None else participants)
    chunk = max(1, int(MAX_USERS_PER_LAUNCH // max(n_items, 1)))

    G = np.zeros((n_items, n_items), dtype=np.float32)
    worker_id = 0
    for I in range(b_m):
        for J in range(b_n):
            if worker_id in live:
                block = np.zeros((rb[I + 1] - rb[I], cb[J + 1] - cb[J]), dtype=np.float32)
                for u0 in range(0, n_users, chunk):
                    u1 = min(u0 + chunk, n_users)
                    A_I = np.ascontiguousarray(R[u0:u1, rb[I]:rb[I + 1]].T)
                    B_J = np.ascontiguousarray(R[u0:u1, cb[J]:cb[J + 1]])
                    block += matmul_launcher(device_ids[worker_id], A_I, B_J, _SRC)
                G[rb[I]:rb[I + 1], cb[J]:cb[J + 1]] = block
            worker_id += 1
    return G


def gram_bytes(n_items, n_interactions, w):
    """Network bytes for the blocked Gram, in both directions (section 2.4.3).

    Out: each item block of R is needed by every worker in its grid row and again
    by every worker in its grid column, so the coordinator ships b_m + b_n copies
    of the dataset (six for a 2 by 4 grid). R is counted as a coordinate list,
    two 4-byte numbers per interaction.

    Back: the blocks are disjoint, so exactly one copy of G returns.
    """
    b_m, b_n = choose_worker_grid(w, n_items, n_items)
    raw = int(n_interactions) * 2 * BYTES_PER_FLOAT
    out = (b_m + b_n) * raw
    back = int(n_items) * int(n_items) * BYTES_PER_FLOAT
    return {"grid": (b_m, b_n), "outbound": int(out), "inbound": int(back),
            "total": int(out + back)}


def recommend(G, liked_items, top_n=10, exclude_seen=True):
    """Item-item recommendations from the exact model.

    Scores are cosine-normalized co-occurrences: G[i, j] / sqrt(G[i, i] G[j, j]),
    summed over the items the user already liked.
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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"CUDA kernel path live: {gpu_available()}\n")
    for (m, n, p), (b_m, b_n) in [((1000, 700, 500), (2, 2)), ((129, 257, 64), (2, 4))]:
        A = rng.standard_normal((m, p)).astype(np.float32)
        B = rng.standard_normal((p, n)).astype(np.float32)
        C = blocked_matmul(A, B, device_ids=[0] * (b_m * b_n), b_m=b_m, b_n=b_n)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        print(f"  m={m:5} n={n:4} p={p:4}  grid {b_m} x {b_n}:  max rel err "
              f"{np.abs(C - ref).max() / np.abs(ref).max():.2e}")
