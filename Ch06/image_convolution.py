"""Chapter 6, section 6.1.1: convolving one image with one filter on the GPU.

    conv2d_basic(device_id, image, filt, _SRC)    listing 6.2. Launch listing 6.1.
    conv2d_const(device_id, image, filt, _SRC, tiled=False, block=(16, 16))
                                                  listing 6.5. Copy the filter into
                                                  constant memory and launch
                                                  listing 6.3, or listing 6.4 with
                                                  shared memory sized for the block
                                                  and its halo.

The filter side is odd, 2r + 1, and r is at most 4 (MAX_RADIUS in kernels.cu).
"""

from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

_SRC = Path(__file__).with_name("kernels.cu").read_text()


def conv2d_basic(device_id, image, filt, _SRC):
    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        _conv = _MODULE.get_function("Conv2DBasicKernel")
        N = cp.asarray(image, dtype=cp.float32)
        F = cp.asarray(filt, dtype=cp.float32)
        P = cp.empty_like(N)
        height, width = N.shape
        r = F.shape[0] // 2
        block = (16, 16)
        grid = ((width + 15) // 16, (height + 15) // 16)
        _conv(grid, block, (N, F, P, np.int32(height), np.int32(width),
                            np.int32(r)))
        return P


def conv2d_const(device_id, image, filt, _SRC, tiled=False, block=(16, 16)):
    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        symbol = _MODULE.get_global("FILTER")
        F = cp.ndarray((filt.size,), dtype=cp.float32, memptr=symbol)
        F[...] = cp.asarray(filt, dtype=cp.float32).ravel()
        name = "Conv2DTiledKernel" if tiled else "Conv2DConstKernel"
        _conv = _MODULE.get_function(name)
        N = cp.asarray(image, dtype=cp.float32)
        P = cp.empty_like(N)
        height, width = N.shape
        r = filt.shape[0] // 2
        grid = ((width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1])
        shared = (block[0] + 2 * r) * (block[1] + 2 * r) * 4 if tiled else 0
        _conv(grid, block, (N, P, np.int32(height), np.int32(width),
                            np.int32(r)), shared_mem=shared)
        return P


if __name__ == "__main__":
    # The worked example of section 6.1: x = [2, 1, 4, 3, 5, 0, 2], f = [1, 2, 1].
    x = np.array([[2, 1, 4, 3, 5, 0, 2]], dtype=np.float32)
    f = np.array([[0, 0, 0], [1, 2, 1], [0, 0, 0]], dtype=np.float32)
    print("y =", cp.asnumpy(conv2d_basic(0, x, f, _SRC))[0].astype(int).tolist())

    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (512, 512)).astype(np.float32)
    filt = rng.normal(0, 1, (5, 5)).astype(np.float32)
    a = conv2d_basic(0, image, filt, _SRC)
    b = conv2d_const(0, image, filt, _SRC)
    c = conv2d_const(0, image, filt, _SRC, tiled=True)
    print("constant memory vs basic, max difference:", float(cp.abs(b - a).max()))
    print("tiled vs basic, max difference          :", float(cp.abs(c - a).max()))
