"""Chapter 2, section 2.3.2: the DeMM coordinator.

    demm_matmul(A, B, device_ids, q, s, h)   listing 2.9. Split the shared dimension
                                             p into one shard per worker, have each
                                             worker return the factors L_r and
                                             M_r^T, and sum the approximate partial
                                             products for validation.

The book writes listing 2.9 as a program with `device_ids = [0, 1]` and
`q, s, h = 32, 10, 2` at module level. Here the same statements are the body of
`demm_matmul`, with those values as defaults, and it returns the factors and the
assembled approximation. Device IDs may repeat on a one-GPU machine.

    python demm_coordinator.py     # compares C_hat with the exact product
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cupy_or_numpy import gpu_available
from demm_launcher import demm_launcher

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(_HERE, "matmulkernel.cu"),
    encoding="utf-8",
) as _f:
    _SRC = _f.read()


def demm_matmul(A, B, device_ids=(0, 1), q=32, s=10, h=2, _SRC=_SRC):
    m = A.shape[0]
    p = A.shape[1]
    n = B.shape[1]

    boundaries = np.linspace(
        0,
        p,
        len(device_ids) + 1,
        dtype=int,
    )

    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = []

        for worker_id, device_id in enumerate(device_ids):
            s_r = boundaries[worker_id]
            e_r = boundaries[worker_id + 1]

            # Create slabs.
            A_r = A[:, s_r:e_r]
            B_r = B[s_r:e_r, :]

            futures.append(
                executor.submit(
                    demm_launcher,
                    device_id,
                    A_r,
                    B_r,
                    q,
                    s,
                    h,
                    _SRC,
                    1000 + worker_id,
                )
            )

        summaries = [future.result() for future in futures]

    # Reconstruct and merge the approximate partial products.
    C_hat = np.zeros((m, n), dtype=np.float32)

    for L_r, M_r_T in summaries:
        C_hat += L_r @ M_r_T

    return summaries, C_hat


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"CUDA kernel path live: {gpu_available()}\n")
    m, p, n, k = 600, 800, 500, 24
    A = (rng.standard_normal((m, k)) @ rng.standard_normal((k, p))).astype(np.float32)
    B = rng.standard_normal((p, n)).astype(np.float32)
    C = A.astype(np.float64) @ B.astype(np.float64)
    for q in (8, 16, 32):
        summaries, C_hat = demm_matmul(A, B, device_ids=[0, 0], q=q)
        sent = sum(L.nbytes + MT.nbytes for L, MT in summaries)
        print(f"  q={q:3}: relative error {np.linalg.norm(C_hat - C) / np.linalg.norm(C):.2e}, "
              f"{sent / 1e6:.2f} MB returned instead of {2 * m * n * 4 / 1e6:.2f} MB")
