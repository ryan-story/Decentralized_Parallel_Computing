"""Chapter 2, sections 2.2.2 and 2.3.2: launching MatrixMulKernel.

    matmul_launcher(device_id, A, B, _SRC)   listing 2.6. One worker's whole job in
                                             the blocked multiplication: compile the
                                             kernel on its device, multiply, return
                                             the result to the host.
    matrix_mul(A, B, _matmul)                section 2.3.2. The same launch with the
                                             result left on the device, so it can feed
                                             straight into another multiplication.
                                             The DeMM worker calls it repeatedly.

    load_kernel_source()                     reads matmulkernel.cu into the string the
                                             listings call _SRC.

The function bodies are the listings'. `cp` comes from `cupy_or_numpy`, which is
CuPy on a GPU machine and a NumPy stand-in otherwise, so the same bodies run on
either path. `matrix_mul` adds one step the listing does not show: it makes both
operands C-contiguous before the launch, because the kernel reads raw row-major
memory and some of the matrices the DeMM worker passes it are not laid out that way.
"""

from __future__ import annotations

import os

import numpy as np

from cupy_or_numpy import cp, gpu_available, to_host  # noqa: F401  (re-exported)

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.join(_HERE, "matmulkernel.cu")


def load_kernel_source():
    """matmulkernel.cu as a string: the `_SRC` the listings pass around."""
    with open(KERNEL_PATH, encoding="utf-8") as _f:
        return _f.read()


def matmul_launcher(device_id, A, B, _SRC):

    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        _matmul = _MODULE.get_function("MatrixMulKernel")

        A = cp.asarray(A, dtype=cp.float32, order="C")
        B = cp.asarray(B, dtype=cp.float32, order="C")

        m = A.shape[0]
        n = B.shape[1]

        C = cp.empty((m, n), dtype=cp.float32)

        p = A.shape[1]

        block = (16, 16)

        grid = (
            (n + block[0] - 1) // block[0],  # blocks across columns
            (m + block[1] - 1) // block[1],  # blocks across rows
        )

        _matmul(grid, block, (A, B, C, np.int32(m), np.int32(n), np.int32(p)))

        return cp.asnumpy(C)


def matrix_mul(A, B, _matmul):
    # MatrixMulKernel indexes raw row-major memory, so its operands must be
    # C-contiguous. QR returns Q in column-major order, and a column slice such as
    # U_hat_r[:, :q] is not contiguous either; this copies only when needed.
    A = cp.ascontiguousarray(A)
    B = cp.ascontiguousarray(B)
    m = A.shape[0]
    p = A.shape[1]
    n = B.shape[1]

    C = cp.empty((m, n), dtype=cp.float32,)

    block = (16, 16)

    grid = (
        (n + block[0] - 1) // block[0],
        (m + block[1] - 1) // block[1],
        )

    _matmul(grid, block, (A, B, C, np.int32(m), np.int32(n), np.int32(p)))

    return C


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    _SRC = load_kernel_source()
    print("CUDA kernel path live:", gpu_available())
    for m, n, p in [(1000, 700, 500), (17, 33, 9), (256, 256, 256)]:
        A = rng.standard_normal((m, p)).astype(np.float32)
        B = rng.standard_normal((p, n)).astype(np.float32)
        C = matmul_launcher(0, A, B, _SRC)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        print(f"  {m:5} x {p:4} times {p:4} x {n:4}:  max rel err "
              f"{np.abs(C - ref).max() / np.abs(ref).max():.2e}")
