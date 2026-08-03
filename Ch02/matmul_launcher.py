"""Chapter 2, section 2.3.1: blocked matmul.

Two entry points, both of which launch `MatrixMulKernel` from `matmulkernel.cu`:

    matmul_launcher(device_id, A_r, B_r)   one worker's whole job in the blocked
                                           matmul: take a block, multiply it,
                                           return the partial product to the host

    matrix_mul(A, B)                       the same launch, but the result stays
                                           on the device so it can feed straight
                                           into another multiply. Section 2.4.2's
                                           DeMM worker calls this repeatedly.

Deviations from the book's listings, and why:

  * The book passes `_SRC` into `matmul_launcher` and `_matmul` into `matrix_mul`,
    because its listings compile the kernel inside the function. Here the source is
    loaded and compiled once at import, so those plumbing parameters are gone. The
    data parameters are unchanged.
  * `matrix_mul` splits a long contraction into chunks (see MAX_WORK_PER_LAUNCH
    below). The book does not show this, because it is not part of the algorithm:
    it keeps one launch from holding the device long enough to trip a driver reset
    at ml-25m scale. The result is identical.
  * `to_host`, `_grid` and `gpu_available` are helpers the book states in prose.

A note on running without a GPU. Everything in this chapter is arithmetic, so the
same code runs on the central processing unit when CuPy or a device is missing,
and `matrix_mul` falls back to NumPy. The result is identical; only the hardware
differs. `gpu_available()` reports which path you are on, and the app and the
notebook both display it, because a claim about decentralized systems should
never be vague about what actually ran.
"""

from __future__ import annotations

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(_HERE, "matmulkernel.cu")

with open(KERNEL_PATH, encoding="utf-8") as _f:
    KERNEL_SOURCE = _f.read()

BLOCK = (16, 16)

try:
    import cupy as cp
    _MODULE = cp.RawModule(code=KERNEL_SOURCE, options=("--std=c++14",))
    _matmul = _MODULE.get_function("MatrixMulKernel")
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                  # no CuPy, no driver, no device
    cp = None
    _matmul = None
    _GPU = False


def gpu_available() -> bool:
    """True when the CUDA kernel path is live; False when NumPy is standing in."""
    return _GPU


def _grid(m: int, n: int):
    """Blocks needed to cover an m by n output, rounding up.

    Rounding up launches a few threads past the edge of the matrix, which is
    exactly what the kernel's boundary guard is there to catch.
    """
    return ((n + BLOCK[0] - 1) // BLOCK[0],      # blocks across columns
            (m + BLOCK[1] - 1) // BLOCK[1])      # blocks across rows


# A naive kernel with a long contraction can run for seconds in one launch, and a
# graphics driver will reset a device that stops responding for too long. So a
# single call is split into chunks of the contracted dimension when it would
# otherwise be too big. This is equation 2.6 again, one level further in: the sum
# splits, so the partial products can be accumulated. It changes nothing about the
# result, only how long any one launch holds the device.
MAX_WORK_PER_LAUNCH = 5e10          # multiply-accumulates


def matrix_mul(A, B):
    """C = A @ B, leaving the result where it was computed.

    On the GPU this is a `MatrixMulKernel` launch (or a few, if the contraction is
    long) and the output stays in device memory, which is what lets section 2.4.2
    chain several multiplies together without a round trip to the host between them.
    """
    if not _GPU:
        return np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32)

    # accept host or device arrays: cp.asarray moves a NumPy array over, and
    # ascontiguousarray then guarantees the row-major layout the kernel indexes
    A = cp.ascontiguousarray(cp.asarray(A, dtype=cp.float32))
    B = cp.ascontiguousarray(cp.asarray(B, dtype=cp.float32))
    m, p = A.shape
    n = B.shape[1]
    if B.shape[0] != p:
        raise ValueError(f"inner dimensions disagree: A is {A.shape}, B is {B.shape}")

    chunk = p
    if m * n * p > MAX_WORK_PER_LAUNCH:
        chunk = max(1, int(MAX_WORK_PER_LAUNCH // max(m * n, 1)))

    if chunk >= p:                                   # the ordinary single launch
        C = cp.empty((m, n), dtype=cp.float32)
        _matmul(_grid(m, n), BLOCK, (A, B, C, np.int32(m), np.int32(n), np.int32(p)))
        return C

    C = cp.zeros((m, n), dtype=cp.float32)
    for s in range(0, p, chunk):
        e = min(s + chunk, p)
        A_k = cp.ascontiguousarray(A[:, s:e])
        B_k = cp.ascontiguousarray(B[s:e, :])
        C_k = cp.empty((m, n), dtype=cp.float32)
        _matmul(_grid(m, n), BLOCK,
                (A_k, B_k, C_k, np.int32(m), np.int32(n), np.int32(e - s)))
        C += C_k                                     # accumulate the partials
    return C


def matmul_launcher(device_id, A_r, B_r):
    """One worker's partial product in the blocked matmul, returned to the host.

    `A_r` is the worker's slice of A's columns and `B_r` the matching slice of B's
    rows, so the product is a full-size partial that the coordinator will add into
    the total. Section 2.3.1 derives that decomposition as equations 2.6 to 2.8.

    The `cp.cuda.Device(device_id)` context selects the GPU that every operation
    inside the block runs on: allocation, launch, and copy-back all happen on that
    worker's assigned device, and CuPy restores the previous device on the way out.
    """
    if not _GPU:
        return np.asarray(matrix_mul(A_r, B_r), dtype=np.float32)

    with cp.cuda.Device(device_id):
        C_r = matrix_mul(cp.asarray(A_r, dtype=cp.float32),
                         cp.asarray(B_r, dtype=cp.float32))
        return cp.asnumpy(C_r)                    # copy back to the host


def to_host(X):
    """Bring a result back to NumPy whichever path produced it."""
    if cp is not None and isinstance(X, cp.ndarray):
        return cp.asnumpy(X)
    return np.asarray(X)
